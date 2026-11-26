"""What a good training set covers, as slots the page fills one picture at a time.

Every slot after the anchor is an edit of the reference picture (<Picture 1>), so the person or object stays the same
while the angle, framing, expression, pose, place and light change. That variety is what lets a LoRA learn the subject
and not the backdrop it was photographed against. A style set is the other way round: many different subjects, all in
the look of the reference.
"""

KINDS = ('character', 'object', 'style')
ASPECTS = {'3:4': (1536, 2048), '1:1': (2048, 2048), '4:3': (2048, 1536), '16:9': (2368, 1344), '9:16': (1344, 2368)}

_C = 'the same person as in <Picture 1>'
_KEEP_C = ('Keep exactly the same face, facial features, skin tone, age, hairstyle, hair colour and body shape as the '
           'person in <Picture 1>. One photograph of one person, sharp focus, no text, no collage.')
_KEEP_CLOTHES = ' Keep the same clothes and accessories as in <Picture 1>.'
_O = 'the same object as in <Picture 1>'
_KEEP_O = ('Keep exactly the same shape, proportions, colours, materials, markings and details as the object in '
           '<Picture 1>. One photograph, sharp focus, no text, no collage.')
_S = 'in exactly the same art style as <Picture 1>'
_KEEP_S = ('Match the medium, line work, brush or render quality, colour palette, lighting mood and texture of '
           '<Picture 1>. Do not copy its subject. No text.')


def _slot(identity, group, label, prompt, aspect='3:4', clothes=True):
    return dict(id=identity, group=group, label=label, prompt=prompt, aspect=aspect, keep_clothes=clothes)


CHARACTER = [
    # Angles
    _slot('front', 'angles', 'Front, full body', f'A full-body photograph of {_C}, standing and facing the camera, arms relaxed, plain light grey studio backdrop, soft even light.'),
    _slot('three_quarter_left', 'angles', 'Three-quarter left', f'A full-body photograph of {_C}, body and head turned 45 degrees to the left, plain light grey studio backdrop, soft even light.'),
    _slot('three_quarter_right', 'angles', 'Three-quarter right', f'A full-body photograph of {_C}, body and head turned 45 degrees to the right, plain light grey studio backdrop, soft even light.'),
    _slot('profile_left', 'angles', 'Side profile', f'A full-body photograph of {_C} in exact side profile facing the left edge of the picture, one eye visible, plain light grey studio backdrop.'),
    _slot('back', 'angles', 'From behind', f'A full-body photograph of {_C} seen from directly behind: back of the head, shoulders and back face the camera, plain light grey studio backdrop.'),
    _slot('low_angle', 'angles', 'Low angle', f'A low-angle photograph looking up at {_C}, standing outdoors under an open sky, dramatic perspective.'),
    _slot('high_angle', 'angles', 'High angle', f'A high-angle photograph looking down at {_C}, standing on a city pavement, camera above head height.'),
    # Close-ups
    _slot('face_front', 'close-ups', 'Face, front', f'A tight head-and-shoulders portrait of {_C}, facing the camera at eye level, neutral expression, soft light, shallow depth of field.', '1:1'),
    _slot('face_three_quarter', 'close-ups', 'Face, three-quarter', f'A close-up portrait of {_C}, head turned three-quarters to the right, eyes toward the camera, soft window light.', '1:1'),
    _slot('face_profile', 'close-ups', 'Face, profile', f'A close-up portrait of {_C} in side profile, clean background, rim light outlining the face.', '1:1'),
    _slot('eyes', 'close-ups', 'Extreme close-up', f'An extreme close-up of the eyes and upper face of {_C}, sharp detail of the eyes, eyebrows and skin.', '16:9'),
    _slot('half_body', 'close-ups', 'Half body', f'A waist-up portrait of {_C}, natural pose, softly blurred indoor background.'),
    # Expressions
    _slot('smile', 'expressions', 'Smiling', f'A portrait of {_C} smiling warmly at the camera.', '1:1'),
    _slot('laugh', 'expressions', 'Laughing', f'A candid portrait of {_C} laughing out loud, eyes creased.', '1:1'),
    _slot('serious', 'expressions', 'Serious', f'A portrait of {_C} with a serious, focused expression, looking straight at the camera.', '1:1'),
    _slot('surprised', 'expressions', 'Surprised', f'A portrait of {_C} looking surprised, eyebrows raised, mouth slightly open.', '1:1'),
    _slot('angry', 'expressions', 'Angry', f'A portrait of {_C} looking angry, brows drawn together.', '1:1'),
    _slot('sad', 'expressions', 'Sad', f'A portrait of {_C} looking sad and thoughtful, eyes lowered.', '1:1'),
    # Poses
    _slot('sitting', 'poses', 'Sitting', f'A full-body photograph of {_C} sitting on a wooden chair in a bright room.'),
    _slot('walking', 'poses', 'Walking', f'A full-body photograph of {_C} walking toward the camera along a city street, mid-stride.'),
    _slot('arms_crossed', 'poses', 'Arms crossed', f'A three-quarter-length photograph of {_C} standing with arms crossed, confident pose, office background.'),
    _slot('running', 'poses', 'Running', f'A dynamic full-body photograph of {_C} running in a park, motion in the hair and clothes.', '4:3'),
    _slot('leaning', 'poses', 'Leaning', f'A photograph of {_C} leaning against a brick wall, relaxed pose.'),
    _slot('waving', 'poses', 'Waving', f'A photograph of {_C} waving at the camera with one hand raised, outdoors.'),
    # Places and light
    _slot('cafe', 'places & light', 'Café, warm light', f'A photograph of {_C} sitting at a table in a cosy café, warm evening light.', '4:3'),
    _slot('golden_hour', 'places & light', 'Golden hour', f'A photograph of {_C} on a beach at golden hour, warm low sunlight from the side.', '4:3'),
    _slot('night_neon', 'places & light', 'Night, neon', f'A photograph of {_C} on a city street at night lit by colourful neon signs.', '3:4'),
    _slot('overcast', 'places & light', 'Overcast forest', f'A photograph of {_C} on a forest path on an overcast day, soft diffuse light.', '3:4'),
    _slot('new_outfit', 'places & light', 'Different outfit', f'A full-body photograph of {_C} wearing a completely different casual outfit, in a living room.', '3:4', clothes=False),
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
    keep = KEEP[kind]
    if kind == 'character' and slot.get('keep_clothes', True) and keep_clothes:
        keep += _KEEP_CLOTHES
    if description and kind != 'style':
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
