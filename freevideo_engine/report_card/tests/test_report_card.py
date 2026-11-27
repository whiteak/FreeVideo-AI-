"""Report Card checks that run without ComfyUI, a GPU or insightface: python -m unittest freevideo_engine.report_card.tests.test_report_card"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from freevideo_engine import report_card as rc


def picture(seed):
    return np.random.default_rng(seed).uniform(40, 200, (36, 64)).astype(np.float32)


def scene(seed, n, drift=0.):
    """n frames of one picture with a little grain, slowly dissolving into another one when drift is set."""
    rng = np.random.default_rng(seed + 100)
    start, end = picture(seed), picture(seed + 50)
    return np.stack([start + (end - start) * min(1., i * drift / 10) + rng.normal(0, 1.5, start.shape) for i in range(n)])


class FrameFlags(unittest.TestCase):
    def checks(self, frames):
        return [f['check'] for f in rc.frame_flags(frames, 24.)]

    def test_clean_take_has_no_flags(self):
        self.assertEqual(self.checks(scene(1, 48, drift=0.1)), [])

    def test_frame_zero_pop(self):
        frames = scene(1, 48, drift=0.1)
        frames[0] = scene(2, 1)[0]
        self.assertEqual(self.checks(frames), ['pop'])

    def test_cut_mid_take_and_last_frame_pop(self):
        frames = np.concatenate([scene(1, 30, 0.1), scene(3, 30, 0.1)])
        frames[-1] = scene(4, 1)[0]
        flags = rc.frame_flags(frames, 24.)
        self.assertEqual([f['check'] for f in flags], ['end_pop', 'cut'])
        self.assertEqual(flags[1]['frame'], 30)

    def test_flicker(self):
        frames = scene(1, 48, drift=0.1)
        frames[20] += 12
        self.assertEqual(self.checks(frames), ['flicker'])

    def test_back_to_start(self):
        frames = scene(1, 60, drift=.3)         # dissolves away from the opening picture...
        frames[45:] = frames[:15]               # ...and is suddenly back
        self.assertEqual(rc.return_to_start(frames), 45)


class Motion(unittest.TestCase):
    def test_reversal(self):
        self.assertIsNone(rc.reversal([0, .1, .2, .3, .35], .15))
        self.assertEqual(rc.reversal([0, .1, .2, .3, .2, .1], .15), 3)
        self.assertEqual(rc.reversal([0, -.2, -.05], .15), 1)

    def test_walk_back_split_at_cuts(self):
        track = [dict(t=float(t), x=500, size=s) for t, s in enumerate([100, 130, 170, 220, 100, 130, 170])]
        self.assertEqual([f['check'] for f in rc.motion_flags({'A': track}, 1000)], ['walk_back'])
        self.assertEqual(rc.motion_flags({'A': track}, 1000, cuts=[4.]), [])

    def test_assign_keeps_one_face_per_card_and_drops_strangers(self):
        samples = [dict(t=0., faces=[dict(name='A', score=.8, box=[0, 0, 100, 100]),
                                     dict(name='A', score=.5, box=[200, 0, 300, 100]),
                                     dict(name='A', score=.1, box=[400, 0, 500, 100]),
                                     dict(name='A', score=.9, box=[0, 0, 40, 40])])]
        tracks, small = rc.assign(samples, ['A'])
        self.assertEqual([p['score'] for p in tracks['A']], [.8])
        self.assertEqual(small, 1)

    def test_face_summary(self):
        card = dict(name='A', bar=.72)
        summary = rc.face_summary(card, [dict(score=s) for s in (.6, .75, .8)])
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['median'], .75)
        self.assertIsNone(rc.face_summary(card, [])['passed'])


class Cards(unittest.TestCase):
    def test_cards_found_from_reference_path_prompt_and_lora(self):
        with tempfile.TemporaryDirectory() as root:
            out, inp = Path(root, 'out'), Path(root, 'in')
            store = out / 'FreeVideo' / '.agent' / 'jobs' / 'cast'
            store.mkdir(parents=True)
            (inp / 'refs').mkdir(parents=True)
            (inp / 'refs' / 'face.png').write_bytes(b'x')
            (inp / 'lora-studio').mkdir()
            for identity, name in (('c1', 'Ada'), ('c2', 'Ben'), ('c3', 'Cal'), ('c4', 'Dee')):
                card = dict(id=identity, kind='character', name=name, images=[
                    dict(media=dict(file='refs/face.png'), label='face'), dict(media=dict(file='refs/face.png'), label='back')])
                (store / (identity + '.json')).write_text(json.dumps(dict(card=card)))
            (inp / 'lora-studio' / 'links.json').write_text(json.dumps({'c3': dict(lora='lora-studio\\cal.safetensors')}))
            take = out / 'FreeVideo' / '2026-01-01' / ('0' * 32)
            take.mkdir(parents=True)
            (take / 'media.json').write_text(json.dumps(dict(references=[dict(path='C:\\in\\freevideo\\agent\\c1\\s.png')],
                                                             loras=[dict(path='C:\\models\\loras\\lora-studio\\cal.safetensors')])))
            (take / 'prompt.txt').write_text('<Subject 1> is Ben: a man.')
            cards = rc.cast_cards(out, inp, take)
            self.assertEqual(sorted(c['name'] for c in cards), ['Ada', 'Ben', 'Cal'])
            self.assertEqual(len(cards[0]['pictures']), 1)   # the back view is left out
            self.assertEqual(cards[0]['bar'], rc.DEFAULT_BAR)
            self.assertEqual([c['name'] for c in rc.cast_cards(out, inp, take, ['c4'])][0], 'Dee')


class Look(unittest.TestCase):
    def test_frames_skip_the_pops(self):
        from freevideo_engine.report_card.look import frame_times
        self.assertEqual(frame_times(362, 24.), [3, 122, 241, 360])

    def test_questions_and_answers(self):
        from freevideo_engine.prompt_vlm.rules import ask_request
        from freevideo_engine.report_card.look import questions, read_answers
        cards = [dict(id='c1', name='Ada', description='a red dress', _asked=True), dict(id='c2', name='Ben', description='')]
        request = questions(['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'], cards, {'c2': dict(prompt_suffix='a black suit')})
        ask_request(request)   # the picture reader accepts it as is
        self.assertEqual(len(request['questions']), 5)
        cards[1]['_asked'] = True
        answers = [dict(text='{"counts": [2, 2, 3, 2]}'), dict(text='```json\n{"problems": []}\n```'),
                   dict(text='{"changes": [{"frame": 2, "what": "floor turns white"}]}'),
                   dict(text='{"differences": []}'), dict(text='not json')]
        flags, notes = read_answers(request, answers, [3, 122, 241, 360], 24., cards)
        self.assertEqual([f['check'] for f in flags], ['people', 'changes'])
        self.assertIn('5.1 s', flags[1]['text'])
        self.assertEqual(len(notes), 1)


class Speech(unittest.TestCase):
    def test_lines_heard_and_digits(self):
        from freevideo_engine.report_card.speech import line_flags
        words = [dict(text=w, start=i * .4, end=i * .4 + .3) for i, w in enumerate('Welcome to CiviSins Entertainment.'.split())]
        flags = line_flags(['Welcome to 7 Sins Entertainment.'], words)
        self.assertEqual([f['check'] for f in flags], ['digits', 'line'])
        good = [dict(text=w, start=i * .4, end=i * .4 + .3) for i, w in enumerate('Welcome to Seven Sins Entertainment.'.split())]
        self.assertEqual(line_flags(['Welcome to Seven Sins Entertainment.'], good), [])
        self.assertEqual([f['check'] for f in line_flags(['Hello there.'], [])], ['line'])

    def test_lip_lag(self):
        from freevideo_engine.report_card.speech import lip_lag, sync_flags, voice_span
        rng = np.random.default_rng(0)
        mouth = np.convolve(rng.random(120), np.ones(3) / 3, 'same')
        voice = np.roll(mouth, 3)                        # the voice three frames after the lips
        self.assertEqual(lip_lag(list(mouth), voice)[0], 3)
        flags, _ = sync_flags((3, .9), 24.)
        self.assertEqual(flags[0]['ms'], 125)
        self.assertIn('earlier', flags[0]['text'])
        self.assertEqual(sync_flags((1, .9), 24.)[0], [])
        self.assertIsNone(lip_lag([None] * 60, voice))
        self.assertEqual(voice_span([dict(start=1., end=4.)], 24., 200), [16, 103])
        self.assertIsNone(voice_span([], 24., 200))


class BestTake(unittest.TestCase):
    def test_take_qa_part_and_score(self):
        weak = dict(passed=False, faces=[dict(name='Ada', median=.6, bar=.72, passed=False)],
                    flags=[dict(check='pop', text='Frame-0 pop.'), dict(check='walk_back', text='Ada walks back.')])
        problems, notes = rc.take_qa_part(weak)
        self.assertEqual(problems, ["Ada's face 0.60, under 0.72", 'Ada walks back'])
        self.assertEqual(notes, ['Frame-0 pop'])
        good = dict(passed=True, faces=[dict(name='Ada', median=.8, bar=.72, passed=True)], flags=[])
        close = dict(passed=False, faces=[dict(name='Ada', median=.9, bar=.72, passed=True)], flags=[dict(check='pop', text='p')])
        ranked = sorted([weak, close, good], key=rc.score, reverse=True)
        self.assertEqual(ranked, [good, close, weak])


class Continuity(unittest.TestCase):
    def test_skin_colour(self):
        from freevideo_engine.report_card.continuity import skin_colour
        frame = np.zeros((60, 60, 3), np.uint8)
        frame[:30] = (200, 140, 110)                     # warm skin tone on top, black below
        sat, luma = skin_colour([frame])
        self.assertAlmostEqual(sat, 0.45, places=2)
        self.assertLess(luma, 80)
        self.assertIsNone(skin_colour([np.zeros((60, 60, 3), np.uint8)])[0])

    def test_colour_flags(self):
        from freevideo_engine.report_card.continuity import colour_flags
        shots = [dict(sat=.5, luma=50.), dict(sat=.51, luma=52.), dict(sat=.4, luma=49.), dict(sat=None, luma=90.)]
        self.assertEqual([(f['check'], f['shot']) for f in colour_flags(shots)], [('colour', 3), ('brightness', 4)])


class LongTake(unittest.TestCase):
    def test_long_take_notes(self):
        from freevideo_engine.chain_plan import long_take_notes
        self.assertEqual(long_take_notes(15.), [])
        self.assertEqual(len(long_take_notes(29.)), 1)
        self.assertEqual(len(long_take_notes(29., last_frame=True)), 2)
        self.assertEqual(long_take_notes(29., last_frame=True, parts=2), [])


if __name__ == '__main__':
    unittest.main()
