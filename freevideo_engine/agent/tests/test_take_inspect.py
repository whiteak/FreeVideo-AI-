"""Tests for the Take Inspector checks (python -m unittest freevideo_engine.agent.tests.test_take_inspect)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from freevideo_engine.agent import take_inspect as ti

RNG = np.random.default_rng(7)
SCENE = RNG.uniform(0, 255, (120, 200)).astype(np.float32)
for _ in range(25):  # smooth it so a shift moves visible shapes, as in a real picture
    SCENE = (SCENE + np.roll(SCENE, 1, 0) + np.roll(SCENE, 1, 1) + np.roll(SCENE, -1, 0) + np.roll(SCENE, -1, 1)) / 5
SCENE = (SCENE - SCENE.min()) / (SCENE.max() - SCENE.min()) * 255


def view(x, y=20, width=64, height=36):
    """A 64x36 window on the scene at (x, y): moving the window is a camera move."""
    x0, y0 = int(round(x)), int(round(y))
    return SCENE[y0:y0 + height, x0:x0 + width].copy()


def clock(count, fps=24):
    return [i / fps for i in range(count)]


class OpeningTest(unittest.TestCase):
    def test_held_frame_passes(self):
        small = [view(10 + i * 0.1) for i in range(48)]
        self.assertTrue(ti.opening(small[0], small, clock(48))['held'])

    def test_jump_after_frame_zero_is_found(self):
        small = [view(10)] + [view(120, 60) for _ in range(47)]
        found = ti.opening(view(10), small, clock(48))
        self.assertFalse(found['held'])
        self.assertAlmostEqual(found['left_at'], 1 / 24, places=2)

    def test_fast_pan_is_not_a_jump(self):
        small = [view(10 + i * 1.5) for i in range(48)]
        self.assertTrue(ti.opening(small[0], small, clock(48))['held'])

    def test_planned_cut_is_allowed(self):
        small = [view(10)] * 12 + [view(120, 60)] * 36
        self.assertTrue(ti.opening(view(10), small, clock(48), planned=[0.5])['held'])

    def test_no_given_frame(self):
        self.assertIsNone(ti.opening(None, [view(0)], [0.0]))


class CameraTest(unittest.TestCase):
    def test_shift_direction(self):
        dx, dy = ti._shift(view(10), view(13))
        self.assertAlmostEqual(dx, -3, delta=0.6)
        self.assertAlmostEqual(dy, 0, delta=0.6)

    def test_pan_one_way_is_fine(self):
        small = [view(10 + i * 1.0) for i in range(96)]
        self.assertEqual(ti.reversals(ti.camera(small, clock(96))), [])

    def test_pan_and_back_is_found(self):
        xs = [10 + i * 1.0 for i in range(48)] + [57 - i * 1.0 for i in range(48)]
        small = [view(x) for x in xs]
        found = ti.reversals(ti.camera(small, clock(96)))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['axis'], 'sideways')


class ReturnTest(unittest.TestCase):
    def test_forward_action_passes(self):
        small = [view(10 + i * 1.2) for i in range(96)]
        self.assertEqual(ti.returns(small, clock(96)), [])

    def test_restart_is_found(self):
        xs = [10 + i * 1.5 for i in range(60)] + [10 + i * 1.5 for i in range(36)]
        found = ti.returns([view(x) for x in xs], clock(96))
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]['at'], 2.5, delta=0.3)
        self.assertAlmostEqual(found[0]['back_to'], 0.0, delta=0.3)

    def test_still_talking_shot_passes(self):
        small = [view(10) + RNG.normal(0, 2, (36, 64)).astype(np.float32) for _ in range(96)]
        self.assertEqual(ti.returns(small, clock(96)), [])

    def test_planned_shot_reverse_shot_passes(self):
        small = [view(10)] * 30 + [view(120, 60)] * 30 + [view(10)] * 36
        self.assertEqual(ti.returns(small, clock(96), planned=[1.25, 2.5]), [])


class FaceTest(unittest.TestCase):
    def test_match_and_mismatch(self):
        her, him, other = np.eye(3)
        frames = [(i * 0.5, i) for i in range(6)]

        def embed(index):  # her face drifts to someone else in the last two frames; his holds
            return [her if index < 4 else other * 0.9 + her * 0.1, him]

        found = ti.faces(None, {'Scarlett': [her], 'James': [him]}, embed, frames=frames)
        self.assertEqual(found['James']['seen'], 6)
        self.assertEqual(found['James']['median'], 1.0)
        self.assertEqual(found['Scarlett']['seen'], 4)  # the drifted frames are a different person
        problems, notes = ti.face_verdicts(found)
        self.assertEqual(problems, [])

        bad = ti.faces(None, {'Scarlett': [her]}, lambda i: [her * 0.6 + other * 0.8], frames=frames)
        problems, _ = ti.face_verdicts(bad)
        self.assertEqual(len(problems), 1)
        self.assertIn('Scarlett', problems[0])

    def test_each_face_goes_to_one_person(self):
        her, him = np.eye(2)
        found = ti.faces(None, {'Scarlett': [her], 'James': [him]}, lambda i: [her], frames=[(0, 0)])
        self.assertEqual(found['Scarlett']['seen'], 1)
        self.assertEqual(found['James']['seen'], 0)


class LipTest(unittest.TestCase):
    def test_offset_found(self):
        fps, rate, count = 24, 16000, 240
        times = [i / fps for i in range(count)]
        talk = (RNG.uniform(0, 1, count) > 0.5).astype(float)
        talk = np.convolve(talk, np.ones(3) / 3, 'same')
        shift = 4  # the sound comes 4 frames (0.167 s) after the mouth
        loud = np.concatenate([np.zeros(shift), talk[:-shift]])
        samples = np.concatenate([RNG.normal(0, 0.02 + 0.3 * level, rate // fps) for level in loud]).astype(np.float32)
        mouth = list(np.abs(np.diff(np.concatenate([[0], talk]))))
        found = ti.lip_offset(times, mouth, samples, rate)
        self.assertAlmostEqual(found['offset'], -shift / fps, delta=1.01 / fps)
        self.assertGreater(found['trust'], ti.LIP_TRUST)


class RepairTest(unittest.TestCase):
    def test_trim_and_shift(self):
        try:
            import av
        except ImportError:
            self.skipTest('PyAV not installed')
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / 'take.mp4', Path(folder) / 'fixed.mp4'
            with av.open(str(source), 'w') as target:
                video = target.add_stream('libx264', rate=Fraction(24))
                video.width, video.height, video.pix_fmt = 64, 48, 'yuv420p'
                voice = target.add_stream('aac', rate=48000, layout='stereo')
                for i in range(48):
                    picture = np.full((48, 64, 3), (i * 5) % 255, np.uint8)
                    for packet in video.encode(av.VideoFrame.from_ndarray(picture, format='rgb24')):
                        target.mux(packet)
                for packet in video.encode():
                    target.mux(packet)
                tone = (0.2 * np.sin(np.arange(96000) / 48000 * 2 * np.pi * 440)).astype(np.float32)
                for at in range(0, tone.size, 1024):
                    chunk = np.ascontiguousarray(np.stack([tone[at:at + 1024]] * 2))
                    frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout='stereo')
                    frame.sample_rate, frame.pts = 48000, at
                    for packet in voice.encode(frame):
                        target.mux(packet)
                for packet in voice.encode():
                    target.mux(packet)
            seconds = ti.repair(source, out, drop_first=1, drop_last=1, audio_shift=0.1)
            self.assertAlmostEqual(seconds, 46 / 24, places=3)
            with av.open(str(out)) as container:
                frames = sum(1 for _ in container.decode(container.streams.video[0]))
            self.assertEqual(frames, 46)



def write_take(path, pictures, fps=24, sound=True):
    import av
    from fractions import Fraction
    with av.open(str(path), 'w') as target:
        video = target.add_stream('libx264', rate=Fraction(fps))
        video.height, video.width = pictures[0].shape[:2]
        video.pix_fmt = 'yuv420p'
        video.options = {'crf': '10'}
        voice = target.add_stream('aac', rate=48000, layout='stereo') if sound else None
        for picture in pictures:
            for packet in video.encode(av.VideoFrame.from_ndarray(picture, format='rgb24')):
                target.mux(packet)
        for packet in video.encode():
            target.mux(packet)
        if voice is not None:
            count = int(48000 * len(pictures) / fps)
            tone = (0.2 * np.sin(np.arange(count) / 48000 * 2 * np.pi * 220)).astype(np.float32)
            for at in range(0, count, 1024):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(np.stack([tone[at:at + 1024]] * 2)), format='fltp', layout='stereo')
                frame.sample_rate, frame.pts = 48000, at
                for packet in voice.encode(frame):
                    target.mux(packet)
            for packet in voice.encode():
                target.mux(packet)


def big(x, y=20):
    """A 320x180 colour frame of the scene window at (x, y)."""
    from PIL import Image
    small = view(x, y)
    image = Image.fromarray(small.astype(np.uint8)).resize((320, 180), Image.Resampling.BICUBIC)
    return np.asarray(image.convert('RGB'))


class CheckTest(unittest.TestCase):
    """take_qa.check on real encoded takes."""

    def setUp(self):
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest('PyAV not installed')
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def given(self, picture):
        from PIL import Image
        path = self.root / 'first.png'
        Image.fromarray(picture).save(path)
        return path

    def test_good_take_is_clean(self):
        from freevideo_engine.agent import take_qa
        pictures = [big(10 + i * 1.0) for i in range(96)]
        path = self.root / 'good.mp4'
        write_take(path, pictures)
        qa = take_qa.check(path, first=self.given(pictures[0]), planned=[])
        self.assertTrue(qa['ok'], qa['problems'])
        self.assertTrue(qa['opening']['held'])

    def test_dropped_first_frame_is_a_problem_not_a_trim(self):
        from freevideo_engine.agent import take_qa
        pictures = [big(10)] + [big(120 + i * 0.4, 60) for i in range(95)]
        path = self.root / 'dropped.mp4'
        write_take(path, pictures)
        qa = take_qa.check(path, first=self.given(pictures[0]), planned=[])
        self.assertFalse(qa['ok'])
        self.assertTrue(any('first frame is dropped' in p for p in qa['problems']), qa['problems'])
        self.assertFalse(any('odd frame at the start' in n for n in qa['notes']), qa['notes'])

    def test_restart_is_a_problem(self):
        from freevideo_engine.agent import take_qa
        xs = [10 + i * 1.5 for i in range(60)] + [10 + i * 1.5 for i in range(60)]
        path = self.root / 'restart.mp4'
        write_take(path, [big(x) for x in xs])
        qa = take_qa.check(path)
        self.assertTrue(any('action goes back' in p for p in qa['problems']), qa['problems'])


if __name__ == '__main__':
    unittest.main()
