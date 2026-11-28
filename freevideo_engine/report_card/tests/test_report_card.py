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


if __name__ == '__main__':
    unittest.main()
