"""Speech helper (fork addition, plan step 4.3): digits inside spoken lines become words before a video is made.

The video model garbles spoken digits ("Welcome to 7 Sins Entertainment" was heard as "CiviSins"), while the same
line written "Seven Sins" comes out clean. Only quoted lines the prompt introduces as speech are changed
(prompt_vlm.rules.spoken_lines); signs, captions and the rest of the prompt keep their digits.
"""
import re

ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()
# Whole numbers only: no leading zero (phone numbers), money ($5), decimals, times (9:30) or parts of words (3D).
NUMBER = re.compile(r'(?<![\w.$€£:])([1-9]\d{0,3}|0)(?!\w|[.,:]\d)')


def words(n):
    """English words for 0-9999; four-digit numbers from 1100 read as years ("nineteen ninety-nine")."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10 - 2] + ('' if n % 10 == 0 else '-' + ONES[n % 10])
    if 1100 <= n <= 9999 and n % 1000 >= 100 and n // 100 % 10:   # years: 1999, 2026 reads "twenty twenty-six"
        head, tail = divmod(n, 100)
        return words(head) + ' ' + ('hundred' if tail == 0 else ('oh ' + ONES[tail]) if tail < 10 else words(tail))
    if n < 1000:
        head, tail = divmod(n, 100)
        return ONES[head] + ' hundred' + ('' if tail == 0 else ' and ' + words(tail))
    head, tail = divmod(n, 1000)
    return words(head) + ' thousand' + ('' if tail == 0 else (' and ' if tail < 100 else ' ') + words(tail))


def spell_line(line):
    """The line with each standalone whole number of up to four digits written out. Capitalised at a sentence start
    or inside a name ("7 Sins" -> "Seven Sins")."""
    def swap(match):
        text = words(int(match.group(1)))
        before, after = line[:match.start()].rstrip(), line[match.end():].lstrip()
        if not before or before[-1] in '.!?' or after[:1].isupper():
            text = text[0].upper() + text[1:]
        return text
    return NUMBER.sub(swap, line)


def spell_spoken(prompt):
    """(prompt with digits in spoken lines written out, the lines changed as [(before, after)])."""
    from .prompt_vlm.rules import spoken_lines
    if not isinstance(prompt, str) or not re.search(r'\d', prompt):
        return prompt, []
    changed = []
    for line in dict.fromkeys(spoken_lines(prompt)):
        after = spell_line(line)
        if after != line:
            changed.append((line, after))
            prompt = prompt.replace(line, after)
    return prompt, changed
