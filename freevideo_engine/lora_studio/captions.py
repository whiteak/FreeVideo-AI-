"""Training captions and likeness checks, written as questions for the installed prompt model (Qwen3-VL).

A subject LoRA learns what the captions leave out: the trigger word comes to mean everything that is in every picture
but never named. So a character caption names the pose, framing, expression, clothes, place and light, and never the
face, hair or body; an object caption names the setting and never the object's own look. A style caption is the
opposite: it names the content and never the style.
"""
import json
import re

SYSTEM = ('You write short training captions for image LoRA datasets. Answer with the caption only: one line, '
          'plain English, comma-separated phrases, no quotes, no lists, no preface.')

_RULES = {
    'subject_character': (
        'Describe this photo for a training caption in 12 to 35 words. Name the camera angle and framing (for example '
        'full body, close-up, three-quarter view, from behind), the pose and action, the facial expression, '
        '{clothes}the background and place, and the lighting. Call a plain backdrop by the colour you actually see. '
        'Do NOT describe the person\'s face, facial features, eye colour, skin tone, hair colour, hairstyle, age, '
        'ethnicity, body shape or name: refer to them only as "the {cls}". Leave out filler such as "no movement" or '
        '"neutral pose".'),
    'subject_object': (
        'Describe this photo for a training caption in 12 to 35 words. Name the camera angle and framing, what the '
        'object is resting on or who is holding or using it, the background and place, and the lighting. Do NOT '
        'describe the object\'s own colour, shape, material, markings, logo or brand: refer to it only as "the {cls}".'),
    'style': (
        'Describe the content of this picture for a training caption in 15 to 40 words: the subject, what is '
        'happening, the setting, the composition and the time of day. Do NOT mention the art style, medium, '
        'technique, colour palette, rendering, brushwork or artist.'),
}

DRIFT_SYSTEM = 'You compare pictures carefully and answer only with JSON.'
DRIFT_QUESTION = {
    'character': ('Image 1 is the reference person. Is image 2 the same person? Judge only identity: the face, facial '
                  'features, hair, skin tone, age and body; ignore pose, angle, expression, lighting{clothes}. Be strict: '
                  'give 9 or 10 only when the face is clearly visible in both pictures and every feature matches. If the '
                  'face in image 2 is hidden, turned away or too small to judge, give at most 7 and say "face not '
                  'visible". A different person with similar hair or clothes gets 3 or less. '
                  'Return {{"score": <0-10, 10 = certainly the same person>, "differences": "<the main differences, '
                  'under 25 words, or none>"}}.'),
    'object': ('Image 1 is the reference object. Is image 2 the same object? Judge shape, proportions, colours, '
               'materials and markings; ignore angle, background and lighting. Return {{"score": <0-10, 10 = exactly '
               'the same object>, "differences": "<the main differences, under 25 words, or none>"}}.'),
    'style': ('Image 1 shows the reference art style. Is image 2 in the same style? Judge medium, line work, colour '
              'palette, texture and rendering; ignore the subject. Return {{"score": <0-10, 10 = the same style>, '
              '"differences": "<the main differences, under 25 words, or none>"}}.'),
}
DRIFT_LIMIT = 6  # Scores below this are flagged as drifted.

_QUOTES = re.compile(r'^[\s"\'`]+|[\s"\'`]+$')
_PREFACE = re.compile(r'^(?:caption|here is (?:the|a) caption|training caption)\s*[:\-]\s*', re.IGNORECASE)


def mode_key(kind, mode):
    if kind == 'style' or mode == 'style':
        return 'style'
    return 'subject_object' if kind == 'object' else 'subject_character'


def caption_question(kind, mode, class_word, vary_clothes=False):
    key = mode_key(kind, mode)
    # A fixed outfit is part of the character, so it is never named: the trigger word learns it with the face. When the
    # clothes change between pictures they are named, so the trigger word does not take any one outfit.
    clothes = 'the clothes, ' if vary_clothes else ''
    if not vary_clothes and key == 'subject_character':
        return _RULES[key].format(cls=class_word or 'person', clothes='').replace(
            'Do NOT describe', 'Do NOT describe the clothes, or')
    return _RULES[key].format(cls=class_word or ('person' if kind == 'character' else 'object'), clothes=clothes)


def clean(text):
    """One caption line from the model's answer: no quotes, no preface, no trailing full stop, single spaces."""
    text = ' '.join(str(text or '').split())
    text = _PREFACE.sub('', _QUOTES.sub('', text))
    return text.rstrip('.').strip()


def with_trigger(caption, trigger, class_word='', kind='character', mode='subject'):
    """The caption as written to the .txt file: the trigger word first, then the class word for a subject set."""
    caption = clean(caption)
    trigger = (trigger or '').strip()
    if not trigger:
        return caption
    lead = trigger if (mode == 'style' or kind == 'style' or not class_word) else '%s %s' % (trigger, class_word.strip())
    if caption.lower().startswith(trigger.lower()):
        return caption
    # "the woman ..." reads better as "ohwx woman ..." than "ohwx woman, the woman ...".
    if class_word and caption.lower().startswith('the %s' % class_word.lower()):
        return lead + caption[len('the %s' % class_word):]
    return '%s, %s' % (lead, caption) if caption else lead


def drift_question(kind, vary_clothes=False):
    return DRIFT_QUESTION['style' if kind == 'style' else kind].format(
        clothes=' and clothes' if vary_clothes else '')


def parse_drift(answer):
    """{'score': 0-10 or None, 'differences': str} from an ask answer (its 'json' or 'text')."""
    value = answer.get('json') if isinstance(answer, dict) else None
    if value is None:
        text = answer.get('text', '') if isinstance(answer, dict) else str(answer or '')
        match = re.search(r'\{.*\}', text, re.DOTALL)
        try:
            value = json.loads(match.group(0)) if match else None
        except ValueError:
            value = None
        if value is None:
            number = re.search(r'\b(10|[0-9])(?:\.\d+)?\b', text)
            return dict(score=float(number.group(0)) if number else None, differences=clean(text)[:200])
    if not isinstance(value, dict):
        return dict(score=None, differences='')
    score = value.get('score')
    try:
        score = max(0., min(10., float(score)))
    except (TypeError, ValueError):
        score = None
    return dict(score=score, differences=clean(value.get('differences', ''))[:200])


def suggest_trigger(name):
    """A rare token from a name: lower-case letters with the vowels thinned, so it is not an everyday word."""
    letters = re.sub(r'[^a-z]', '', (name or '').lower())
    if not letters:
        return 'sks'
    head, rest = letters[0], re.sub(r'[aeiou]', '', letters[1:])
    token = (head + rest)[:6]
    if len(token) < 3:
        token = (letters + 'xq')[:4]
    return token + 'v'
