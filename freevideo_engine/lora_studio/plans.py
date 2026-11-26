"""What a good training set covers, as slots the page fills one picture at a time.

Every slot after the anchor is an edit of the reference picture (<Picture 1>), so the person or object stays the same
while the angle, framing, expression, pose, place and light change. That variety is what lets a LoRA learn the subject
and not the backdrop it was photographed against. A style set is the other way round: many different subjects, all in
the look of the reference.
"""

KINDS = ('character', 'object', 'style')
ASPECTS = {'3:4': (1536, 2048), '1:1': (2048, 2048), '4:3': (2048, 1536), '16:9': (2368, 1344), '9:16': (1344, 2368)}

_C = 'the same person as in <Picture 1>'
_KEEP_C = ('It is the same person as in <Picture 1>: the same face, facial features, skin tone, age, hairstyle, hair '
           'colour and build. One photograph of one person, sharp focus, no text, no collage.')
_KEEP_CLOTHES = ' Keep the same clothes and accessories as in <Picture 1>.'
_O = 'the same object as in <Picture 1>'
_KEEP_O = ('Keep exactly the same shape, proportions, colours, materials, markings and details as the object in '
           '<Picture 1>. One photograph, sharp focus, no text, no collage.')
_S = 'in exactly the same art style as <Picture 1>'
_KEEP_S = ('Match the medium, line work, brush or render quality, colour palette, lighting mood and texture of '
           '<Picture 1>. Do not copy its subject. No text.')


def _slot(identity, group, label, prompt, aspect='3:4', clothes=True):
    return dict(id=identity, group=group, label=label, prompt=prompt, aspect=aspect, keep_clothes=clothes)


# Character slots are written as edit commands for H3 Image: what to change first, then what to keep. Prompts that
# described the finished photo made H3 return the reference almost unchanged (tested on an RTX 5080, 2026-11-26),
# so each one names the change outright and says the old pose, framing or expression must not be copied. The turns
# reuse the cast cards' measured view words (image_gen.cast_views), which H3 follows.
# "Left" alone was read as the person's own left on Jason's PC (the picture turned to the right of the frame), so the
# turns name the picture's edge and where the nose, chest and toes point, as the measured side view does.
_TURN_45 = ('将<Picture 1>中的同一个人向画面左边转45度，呈现四分之三侧面。鼻尖、胸口、脚尖朝向画面左边（观看者的左边），'
            '两只眼睛都能看见。头和身体一起转。全身站立，双脚入画，浅灰纯色背景。')
_TURN_45R = ('将<Picture 1>中的同一个人向画面右边转45度，呈现四分之三侧面。鼻尖、胸口、脚尖朝向画面右边（观看者的右边），'
             '两只眼睛都能看见。头和身体一起转。全身站立，双脚入画，浅灰纯色背景。')
_SIDE = ('将<Picture 1>中的同一个人完全转向画面左边，呈现90度正侧面。鼻尖、胸口、脚尖朝向画面左边，只能看见一只眼睛，'
         '近侧肩膀遮住远侧肩膀。全身站立，双脚入画，浅灰纯色背景。')
_BACK = ('将<Picture 1>中的同一个人整个身体转过180度，完全背对镜头。头、肩膀、躯干、臀部和双脚一起转过去，后脑勺、背部和'
         '脚后跟朝向镜头。看不到脸。全身，浅灰纯色背景。')
_NEW = ' Do not copy the pose, framing or expression of <Picture 1>; only the person stays the same.'
_FACE = (' Reframe as a close portrait: crop from the top of the head to the shoulders, the face fills about half of '
         'the picture.')


def _edit(identity, group, label, change, aspect='3:4', clothes=True, new=True):
    return _slot(identity, group, label, change + (_NEW if new else ''), aspect, clothes)


CHARACTER = [
    # Angles
    _edit('front', 'angles', 'Front, full body', 'Show the person in <Picture 1> standing straight and facing the camera, the whole body from head to feet, arms relaxed at the sides, plain light grey studio backdrop, soft even light.', new=False),
    _edit('three_quarter_left', 'angles', 'Three-quarter left', _TURN_45, new=False),
    _edit('three_quarter_right', 'angles', 'Three-quarter right', _TURN_45R, new=False),
    _edit('profile_left', 'angles', 'Side profile', _SIDE, new=False),
    _edit('back', 'angles', 'From behind', _BACK, new=False),
    _edit('low_angle', 'angles', 'Low angle', 'Move the camera down to knee height and tilt it up at the person in <Picture 1>, who stands outdoors under an open blue sky. Strong upward perspective, the sky fills the background.'),
    _edit('high_angle', 'angles', 'High angle', 'Move the camera high above the person in <Picture 1> and look down at them standing on a city pavement; the top of the head is closest to the camera.'),
    # Close-ups
    _edit('face_front', 'close-ups', 'Face, front', 'Make a tight close-up portrait of the face of the person in <Picture 1>, facing the camera at eye level, neutral expression, soft light, blurred background.' + _FACE, '1:1', new=False),
    _edit('face_three_quarter', 'close-ups', 'Face, three-quarter', 'Make a close-up portrait of the person in <Picture 1> with the head turned three-quarters so the nose points toward the right edge of the picture (seen from the camera) and the eyes looking at the camera, soft window light.' + _FACE, '1:1'),
    _edit('face_profile', 'close-ups', 'Face, profile', 'Make a close-up portrait of the person in <Picture 1> in exact side profile facing the left edge, one eye visible, rim light outlining the face.' + _FACE, '1:1'),
    _edit('eyes', 'close-ups', 'Extreme close-up', 'Make an extreme close-up of the eyes and upper face of the person in <Picture 1>: the eyes and eyebrows fill the frame, sharp skin detail.', '16:9'),
    _edit('half_body', 'close-ups', 'Half body', 'Reframe <Picture 1> as a waist-up portrait of the same person, one hand in a pocket, turned slightly to the side, softly blurred indoor background.'),
    # Expressions
    _edit('smile', 'expressions', 'Smiling', 'Change the expression of the person in <Picture 1> to a big warm smile showing the teeth, eyes crinkled. Reframe as a head-and-shoulders portrait.', '1:1'),
    _edit('laugh', 'expressions', 'Laughing', 'Make the person in <Picture 1> laugh out loud with the mouth wide open and the head tilted back slightly. Head-and-shoulders portrait.', '1:1'),
    _edit('serious', 'expressions', 'Serious', 'Give the person in <Picture 1> a serious, intense expression: brows lowered, lips pressed together, staring into the camera. Head-and-shoulders portrait, dramatic side light.', '1:1'),
    _edit('surprised', 'expressions', 'Surprised', 'Make the person in <Picture 1> look shocked: eyebrows raised high, eyes wide, mouth open in an O. Head-and-shoulders portrait.', '1:1'),
    _edit('angry', 'expressions', 'Angry', 'Make the person in <Picture 1> look furious: brows drawn together, teeth clenched, nostrils flared. Head-and-shoulders portrait.', '1:1'),
    _edit('sad', 'expressions', 'Sad', 'Make the person in <Picture 1> look sad: eyes lowered and glistening, mouth turned down. Head-and-shoulders portrait, soft grey light.', '1:1'),
    # Poses
    _edit('sitting', 'poses', 'Sitting', 'Show the person in <Picture 1> sitting on a wooden chair in a bright room, legs crossed, hands resting on the knee, whole body visible.'),
    _edit('walking', 'poses', 'Walking', 'Show the person in <Picture 1> walking toward the camera along a city street in mid-stride, one foot in front of the other, arms swinging.'),
    _edit('arms_crossed', 'poses', 'Arms crossed', 'Show the person in <Picture 1> standing with the arms firmly crossed over the chest and a confident look, in a modern office.'),
    _edit('running', 'poses', 'Running', 'Show the person in <Picture 1> running fast through a park, body leaning forward, one knee high, hair and clothes moving.', '4:3'),
    _edit('leaning', 'poses', 'Leaning', 'Show the person in <Picture 1> leaning one shoulder against a red brick wall, ankles crossed, hands in pockets.'),
    _edit('waving', 'poses', 'Waving', 'Show the person in <Picture 1> waving at the camera with the right hand raised high above the shoulder, smiling, outdoors.'),
    # Places and light
    _edit('cafe', 'places & light', 'Café, warm light', 'Place the person from <Picture 1> at a table in a cosy café holding a cup, warm evening light, medium shot.', '4:3'),
    _edit('golden_hour', 'places & light', 'Golden hour', 'Place the person from <Picture 1> on a beach at golden hour, warm low sunlight from the side, long shadows, medium shot.', '4:3'),
    _edit('night_neon', 'places & light', 'Night, neon', 'Place the person from <Picture 1> on a city street at night lit by pink and blue neon signs, reflections on wet pavement.'),
    _edit('overcast', 'places & light', 'Overcast forest', 'Place the person from <Picture 1> on a forest path on an overcast day, soft diffuse light, trees behind.'),
    _edit('new_outfit', 'places & light', 'Different outfit', 'Dress the person from <Picture 1> in a completely different outfit (a knitted sweater and dark trousers) and show them full body in a living room.', '3:4', clothes=False),
]

OBJECT = [
    # Angles
    _slot('front', 'angles', 'Front', f'A product photograph of {_O} from the front, centred, plain light grey backdrop, soft even studio light.', '1:1'),
    _slot('back', 'angles', 'Back', f'A product photograph of the back of {_O}, centred, plain light grey backdrop, soft even studio light.', '1:1'),
    _slot('left', 'angles', 'Left side', f'A product photograph of {_O} from its left side, plain light grey backdrop.', '1:1'),
    _slot('right', 'angles', 'Right side', f'A product photograph of {_O} from its right side, plain light grey backdrop.', '1:1'),
    _slot('three_quarter', 'angles', 'Three-quarter', f'A product photograph of {_O} from a three-quarter angle above, plain light grey backdrop.', '1:1'),
    _slot('top_down', 'angles', 'Top down', f'A flat-lay photograph of {_O} seen from directly above on a white table.', '1:1'),
    _slot('low_angle', 'angles', 'Low angle', f'A low-angle hero photograph of {_O}, camera close to the ground looking up at it.', '4:3'),
    # Details
    _slot('detail_material', 'details', 'Material close-up', f'A macro close-up of the surface material and texture of {_O}.', '1:1'),
    _slot('detail_feature', 'details', 'Feature close-up', f'A close-up of the most distinctive part or marking of {_O}, sharp detail.', '1:1'),
    _slot('detail_edge', 'details', 'Edge and shape', f'A close-up of the edges and silhouette of {_O} against a dark background, rim light.', '4:3'),
    # Places
    _slot('table', 'places', 'On a table', f'A photograph of {_O} on a wooden kitchen table, natural window light.', '4:3'),
    _slot('outdoors', 'places', 'Outdoors', f'A photograph of {_O} outdoors on a stone wall in a garden, daylight.', '4:3'),
    _slot('shelf', 'places', 'On a shelf', f'A photograph of {_O} on a shelf among other everyday things, indoor light.', '4:3'),
    _slot('street', 'places', 'City street', f'A photograph of {_O} on a city pavement, shallow depth of field.', '4:3'),
    # In use
    _slot('in_hand', 'in use', 'Held in a hand', f'A photograph of a hand holding {_O}, close framing.', '1:1'),
    _slot('in_use', 'in use', 'Being used', f'A photograph of a person using {_O} in the way it is meant to be used, natural setting.', '4:3'),
    _slot('scale', 'in use', 'Next to a person', f'A photograph of {_O} next to a person for a sense of its size.', '3:4'),
    # Light
    _slot('studio_white', 'light', 'Studio white', f'A clean e-commerce photograph of {_O} on pure white, bright even light.', '1:1'),
    _slot('dramatic', 'light', 'Dramatic dark', f'A moody photograph of {_O} on black with a single hard spotlight.', '1:1'),
    _slot('warm', 'light', 'Warm light', f'A photograph of {_O} in warm candle-like light, cosy evening scene.', '4:3'),
    _slot('cool', 'light', 'Cool light', f'A photograph of {_O} in cool blue morning light by a window.', '4:3'),
]

STYLE = [
    _slot('portrait', 'people', 'Portrait', f'A portrait of an elderly fisherman, {_S}.', '3:4'),
    _slot('child', 'people', 'Child playing', f'A child playing with a kite in a field, {_S}.', '4:3'),
    _slot('crowd', 'people', 'Crowd', f'A busy market crowd, {_S}.', '16:9'),
    _slot('dancer', 'people', 'Dancer', f'A dancer mid-leap on a stage, {_S}.', '3:4'),
    _slot('city', 'places', 'City', f'A city skyline at dusk, {_S}.', '16:9'),
    _slot('forest', 'places', 'Forest', f'A misty forest with a stream, {_S}.', '16:9'),
    _slot('interior', 'places', 'Interior', f'A cosy living room interior, {_S}.', '4:3'),
    _slot('mountain', 'places', 'Mountains', f'Snowy mountains under a starry sky, {_S}.', '16:9'),
    _slot('cat', 'animals', 'Cat', f'A cat asleep on a windowsill, {_S}.', '4:3'),
    _slot('horse', 'animals', 'Horse', f'A horse galloping across a beach, {_S}.', '16:9'),
    _slot('bird', 'animals', 'Bird', f'A bird perched on a branch, {_S}.', '1:1'),
    _slot('food', 'things', 'Food', f'A bowl of noodles on a table, {_S}.', '1:1'),
    _slot('car', 'things', 'Car', f'A vintage car parked on a street, {_S}.', '4:3'),
    _slot('still_life', 'things', 'Still life', f'A still life of fruit and a jug, {_S}.', '1:1'),
    _slot('machine', 'things', 'Machine', f'A giant steam-powered robot, {_S}.', '3:4'),
    _slot('night', 'moods', 'Night scene', f'A lantern-lit alley at night in the rain, {_S}.', '9:16'),
    _slot('action', 'moods', 'Action', f'Two sword fighters clashing, {_S}.', '16:9'),
    _slot('calm', 'moods', 'Calm', f'A lone boat on a still lake at dawn, {_S}.', '16:9'),
]

PLANS = {'character': CHARACTER, 'object': OBJECT, 'style': STYLE}
KEEP = {'character': _KEEP_C, 'object': _KEEP_O, 'style': _KEEP_S}

# A first picture made from words alone, when the user has no photo: every other slot edits it.
ANCHOR = {
    'character': ('A full-body photograph of {description}. The person stands alone facing the camera, the whole body '
                  'from head to feet, arms relaxed, plain light grey studio backdrop, soft even light, sharp focus, no text.', '3:4'),
    'object': ('A product photograph of {description}. The single object alone, centred, plain light grey backdrop, '
               'soft even studio light, sharp focus, no people, no text.', '1:1'),
    'style': ('{description}. A lighthouse on a rocky coast.', '4:3'),
}

# Generic pictures of the class, without the subject: they stop the LoRA from turning every person into this person.
REGULARISATION = {
    'character': ['A photograph of a {cls} standing in a park.', 'A portrait of a {cls} smiling, soft light.',
                  'A photograph of a {cls} sitting in a café.', 'A full-body photograph of a {cls} walking on a street.',
                  'A close-up portrait of a {cls}, studio light.', 'A photograph of a {cls} at the beach.',
                  'A photograph of a {cls} in an office.', 'A photograph of a {cls} at night under street lights.'],
    'object': ['A product photograph of a {cls} on a grey backdrop.', 'A photograph of a {cls} on a table.',
               'A photograph of a {cls} outdoors.', 'A close-up photograph of a {cls}.',
               'A photograph of a hand holding a {cls}.', 'A photograph of a {cls} on a shelf.'],
    'style': ['A photograph of a lighthouse.', 'A photograph of a city street.', 'A photograph of a cat.',
              'A photograph of a forest.', 'A photograph of a person reading.', 'A photograph of a bowl of fruit.'],
}


def size(aspect):
    return ASPECTS.get(aspect, ASPECTS['1:1'])


def slot_prompt(kind, slot, description='', keep_clothes=True):
    """The full edit prompt of a slot: the change it asks for plus the words that hold the subject still."""
    text = slot['prompt']
    if kind == 'object' and slot['id'] != 'front':
        text = 'Make a new photograph of the object in <Picture 1>. ' + text + (
            ' Do not copy the camera angle, background or lighting of <Picture 1>; only the object stays the same.')
    keep = KEEP[kind]
    if kind == 'character' and slot.get('keep_clothes', True) and keep_clothes:
        keep += _KEEP_CLOTHES
    if description and kind != 'style' and (kind != 'character' or slot.get('keep_clothes', True)):
        keep += ' The subject: %s.' % description.strip().rstrip('.')[:400]
    return '%s %s' % (text, keep)


def anchor_prompt(kind, description):
    text, aspect = ANCHOR[kind]
    return text.format(description=(description or '').strip().rstrip('.')), aspect


def regularisation_prompts(kind, class_word, count):
    words = (class_word or ('person' if kind == 'character' else 'object')).strip()
    rows = REGULARISATION[kind]
    return [rows[i % len(rows)].format(cls=words) for i in range(max(0, count))]


def page_plan(kind):
    """The slot list the page shows: id, group, label, aspect, width and height."""
    if kind not in PLANS:
        raise ValueError('invalid_kind')
    return [dict(id=row['id'], group=row['group'], label=row['label'], aspect=row['aspect'],
                 width=size(row['aspect'])[0], height=size(row['aspect'])[1]) for row in PLANS[kind]]


def find(kind, slot_id):
    for row in PLANS.get(kind, ()):
        if row['id'] == slot_id:
            return row
    raise ValueError('unknown_slot')
