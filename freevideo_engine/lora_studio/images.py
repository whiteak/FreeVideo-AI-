"""Picture preparation for training sets: open, straighten, crop, shrink, score and fingerprint.

Everything here is Pillow and NumPy on the CPU, so imports and checks never wait for the graphics card.
"""
import io

MAX_SIDE = 2048          # Trainers bucket down to 512-1024; a bigger file only costs disk and load time.
MIN_SIDE = 384           # Below this a picture teaches the model blur.
MAX_INPUT_PIXELS = 80_000_000
BACKDROP = (236, 236, 236)  # The light grey of the engine's cast pictures.
DUPLICATE_DISTANCE = 6   # dHash bits out of 64; 0-6 is the same picture resized, recompressed or lightly edited.
BLURRY = 60.0            # Variance of the Laplacian on a 512 px grey copy; sharp photos score in the hundreds.


def open_image(data_or_path):
    """An RGB Pillow image with its orientation tag applied; transparency is laid on the light grey backdrop."""
    from PIL import Image, ImageOps
    source = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else data_or_path
    try:
        with Image.open(source) as opened:
            if opened.width * opened.height > MAX_INPUT_PIXELS:
                raise ValueError('too_large')
            image = ImageOps.exif_transpose(opened)
            image.load()
    except (OSError, Image.DecompressionBombError, SyntaxError) as error:
        raise ValueError('invalid_media') from error
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        flat = Image.new('RGB', rgba.size, BACKDROP)
        flat.paste(rgba, mask=rgba.getchannel('A'))
        return flat
    return image.convert('RGB')


def trim_borders(image, tolerance=10):
    """Cut flat bars (letterboxing, scanner margins) off the edges; a picture that is all one colour stays."""
    import numpy as np
    pixels = np.asarray(image, dtype=np.int16)
    height, width = pixels.shape[:2]

    def flat(line, colour):
        return np.abs(line - colour).max() <= tolerance

    top, bottom, left, right = 0, height, 0, width
    corner = pixels[0, 0]
    while top < bottom - 1 and flat(pixels[top], corner):
        top += 1
    corner = pixels[-1, -1]
    while bottom - 1 > top and flat(pixels[bottom - 1], corner):
        bottom -= 1
    corner = pixels[0, 0]
    while left < right - 1 and flat(pixels[top:bottom, left], corner):
        left += 1
    corner = pixels[-1, -1]
    while right - 1 > left and flat(pixels[top:bottom, right - 1], corner):
        right -= 1
    # Only real bars: keep the picture when trimming would take more than 40% or less than 2% off a side pair.
    if (bottom - top) < height * .6 or (right - left) < width * .6:
        return image
    if (height - (bottom - top)) < height * .02 and (width - (right - left)) < width * .02:
        return image
    return image.crop((left, top, right, bottom))


def subject_box(width, height, face, kind='character'):
    """A crop box around a person whose face is small in a big photo: head room above, the body below.

    `face` is [left, top, right, bottom]. Returns None when the face already fills enough of the picture."""
    left, top, right, bottom = face
    face_w, face_h = max(1., right - left), max(1., bottom - top)
    if face_w * face_h >= width * height * .02:
        return None  # Already a portrait or a half-length shot.
    centre = (left + right) / 2
    box_w = min(width, face_w * 5)
    box_top = max(0., top - face_h * .8)
    box_bottom = min(float(height), top + face_h * 9)  # About a full body below a head.
    box_left = min(max(0., centre - box_w / 2), width - box_w)
    box = (round(box_left), round(box_top), round(box_left + box_w), round(box_bottom))
    if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
        return None
    return box


def face_crops(image, face):
    """A head-and-shoulders and a half-length crop around `face` ([left, top, right, bottom]), as (label, image) pairs.

    A set of mostly full-body pictures leaves the face a few dozen pixels tall at 512 px training, and the LoRA learns
    the clothes but not the face (Scarlett's first LoRA). Crops that would be smaller than MIN_SIDE are left out."""
    left, top, right, bottom = face
    face_w, face_h = max(1., right - left), max(1., bottom - top)
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    out = []
    for label, width, above, below in (('face', face_w * 2.4, face_h * .9, face_h * 1.5),
                                       ('half', face_w * 4.5, face_h * 1.0, face_h * 5.0)):
        box_w, box_h = min(width, image.width), min(above + below, image.height)
        box_left = min(max(0., centre_x - box_w / 2), image.width - box_w)
        box_top = min(max(0., centre_y - above), image.height - box_h)
        box = (round(box_left), round(box_top), round(box_left + box_w), round(box_top + box_h))
        if min(box[2] - box[0], box[3] - box[1]) < MIN_SIDE:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > image.width * image.height * .7:
            continue  # Already about that framing.
        out.append((label, image.crop(box)))
    return out


def fit(image, max_side=MAX_SIDE):
    from PIL import Image
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def sharpness(image):
    """Variance of the Laplacian on a 512 px grey copy: low means blurred or out of focus."""
    import numpy as np
    grey = image.convert('L')
    grey.thumbnail((512, 512))
    a = np.asarray(grey, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def dhash(image, size=8):
    """64-bit difference hash as 16 hex digits: near-identical pictures differ in a few bits."""
    from PIL import Image
    import numpy as np
    grey = np.asarray(image.convert('L').resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = 0
    for value in (grey[:, :-1] > grey[:, 1:]).flatten():
        bits = (bits << 1) | int(value)
    return '%016x' % bits


def distance(a, b):
    try:
        return bin(int(a, 16) ^ int(b, 16)).count('1')
    except (TypeError, ValueError):
        return 64


def duplicates(rows, limit=DUPLICATE_DISTANCE):
    """Ids of rows that repeat an earlier row: rows are (id, hash) in keep order."""
    kept, repeated = [], []
    for identity, value in rows:
        if any(distance(value, other) <= limit for other in kept):
            repeated.append(identity)
        else:
            kept.append(value)
    return repeated


def quality(image):
    score = sharpness(image)
    return dict(sharpness=round(score, 1), blurry=score < BLURRY, small=min(image.size) < MIN_SIDE)


def prepare(data, kind='character', faces=None, crop=True):
    """An imported photo made ready: straightened, bars trimmed, cropped to the person when they are small in the
    frame (characters, from `faces` boxes found on the original), shrunk to MAX_SIDE. Returns (image, notes)."""
    image = open_image(data)
    notes = []
    if crop:
        trimmed = trim_borders(image)
        if trimmed.size != image.size:
            notes.append('trimmed')
        image = trimmed
        # Face boxes were found on the untrimmed picture, so a trimmed one is not cropped again.
        if kind == 'character' and faces and not notes:
            best = max(faces, key=lambda row: (row['box'][2] - row['box'][0]) * (row['box'][3] - row['box'][1]))
            box = subject_box(image.width, image.height, best['box'])
            if box:
                image = image.crop(box)
                notes.append('cropped_to_person')
            if len(faces) > 1:
                notes.append('several_people')
        elif kind == 'character' and faces == []:
            notes.append('no_face')
    image = fit(image)
    return image, notes


def cutout(image, mask, backdrop=BACKDROP, feather=2):
    """The object kept where the mask is white, the rest replaced by a plain backdrop, then cropped to the object
    with a margin so it fills most of the frame."""
    from PIL import Image, ImageFilter
    mask = mask.convert('L').resize(image.size, Image.NEAREST)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    flat = Image.new('RGB', image.size, backdrop)
    flat.paste(image, mask=mask)
    box = mask.point(lambda v: 255 if v > 127 else 0).getbbox()
    if not box:
        raise ValueError('empty_mask')
    left, top, right, bottom = box
    margin = round(max(right - left, bottom - top) * .12)
    side_w, side_h = right - left + 2 * margin, bottom - top + 2 * margin
    cx, cy = (left + right) / 2, (top + bottom) / 2
    box = (round(cx - side_w / 2), round(cy - side_h / 2), round(cx + side_w / 2), round(cy + side_h / 2))
    canvas = Image.new('RGB', (box[2] - box[0], box[3] - box[1]), backdrop)
    canvas.paste(flat.crop((max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))),
                 (max(0, -box[0]), max(0, -box[1])))
    return canvas


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    image.save(temporary, format='PNG', optimize=False)
    temporary.replace(path)
