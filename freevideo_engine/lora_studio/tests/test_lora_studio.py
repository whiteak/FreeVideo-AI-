"""LoRA Studio checks that run without ComfyUI, a GPU or the models: python -m unittest freevideo_engine.lora_studio.tests.test_lora_studio"""
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from freevideo_engine.lora_studio import captions, cast_loras, export, images, plans, store, training
from freevideo_engine.lora_studio.routes import Context


def picture(seed=0, size=(800, 1000), bars=0):
    image = Image.new('RGB', size, (30 + seed * 20 % 200, 90, 160))
    draw = ImageDraw.Draw(image)
    for i in range(12):
        x = (seed * 37 + i * 71) % size[0]
        draw.ellipse((x, i * 70, x + 120, i * 70 + 90), fill=((seed * 53 + i * 29) % 255, (i * 41) % 255, (seed * 17) % 255))
    if bars:
        canvas = Image.new('RGB', (size[0], size[1] + 2 * bars), (0, 0, 0))
        canvas.paste(image, (0, bars))
        image = canvas
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


class Folders:
    def __init__(self, root):
        self.root = Path(root)
        for name in ('input', 'output', 'models/loras'):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def get_input_directory(self):
        return str(self.root / 'input')

    def get_output_directory(self):
        return str(self.root / 'output')

    def get_folder_paths(self, kind):
        return [str(self.root / 'models' / kind)]

    def get_filename_list(self, kind):
        base = self.root / 'models' / kind
        return sorted(str(p.relative_to(base)).replace('\\', '/') for p in base.rglob('*.safetensors'))

    def get_full_path_or_raise(self, kind, name):
        path = self.root / 'models' / kind / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return str(path)


class ImageTests(unittest.TestCase):
    def test_trim_and_hash(self):
        image, notes = images.prepare(picture(1, bars=120), 'object')
        self.assertIn('trimmed', notes)
        self.assertEqual(image.size, (800, 1000))
        self.assertEqual(images.dhash(image), images.dhash(images.open_image(picture(1))))

    def test_duplicates(self):
        a = images.dhash(images.open_image(picture(1)))
        small = images.open_image(picture(1)).resize((400, 500))
        b, c = images.dhash(small), images.dhash(images.open_image(picture(7)))
        self.assertEqual(images.duplicates([('a', a), ('b', b), ('c', c)]), ['b'])

    def test_subject_box_and_fit(self):
        self.assertIsNone(images.subject_box(1000, 1000, [400, 300, 600, 550]))
        box = images.subject_box(4000, 3000, [1900, 400, 2000, 520])
        self.assertTrue(box[0] < 1900 < 2000 < box[2] and box[1] < 400)
        self.assertEqual(max(images.fit(Image.new('RGB', (5000, 2500))).size), images.MAX_SIDE)

    def test_face_crops(self):
        picture = Image.new('RGB', (1600, 2400), (200, 200, 200))
        crops = dict(images.face_crops(picture, [700, 300, 900, 550]))
        self.assertEqual(set(crops), {'face', 'half'})
        self.assertGreaterEqual(min(crops['face'].size), images.MIN_SIDE)
        self.assertEqual(images.face_crops(picture, [790, 300, 830, 350]), [])  # too small to be worth a crop

    def test_cutout(self):
        image = Image.new('RGB', (400, 400), (255, 0, 0))
        mask = Image.new('L', (400, 400), 0)
        ImageDraw.Draw(mask).rectangle((150, 150, 250, 250), fill=255)
        out = images.cutout(image, mask)
        self.assertLess(out.width, 200)
        self.assertEqual(out.getpixel((2, 2)), images.BACKDROP)
        with self.assertRaises(ValueError):
            images.cutout(image, Image.new('L', (400, 400), 0))


class PlanAndCaptionTests(unittest.TestCase):
    def test_plans(self):
        for kind in plans.KINDS:
            rows = plans.page_plan(kind)
            self.assertGreaterEqual(len(rows), 15)
            self.assertEqual(len({r['id'] for r in rows}), len(rows))
            prompt = plans.slot_prompt(kind, plans.find(kind, rows[0]['id']), 'a red mug')
            self.assertIn('<Picture 1>', prompt)
        self.assertNotIn('same clothes', plans.slot_prompt('character', plans.find('character', 'new_outfit')))
        self.assertEqual(len(plans.regularisation_prompts('character', 'woman', 10)), 10)

    def test_captions(self):
        self.assertEqual(captions.with_trigger('"The woman walking in a park."', 'mya', 'woman'), 'mya woman walking in a park')
        self.assertEqual(captions.with_trigger('Caption: close-up, soft light', 'mya', 'woman'), 'mya woman, close-up, soft light')
        self.assertEqual(captions.with_trigger('a city at night', 'inkw', '', 'style', 'style'), 'inkw, a city at night')
        self.assertEqual(captions.parse_drift({'json': {'score': 8, 'differences': 'none'}})['score'], 8.0)
        self.assertEqual(captions.parse_drift({'text': '```json\n{"score": 3, "differences": "different nose"}\n```'})['score'], 3.0)
        self.assertNotIn('hair', captions.caption_question('object', 'subject', 'mug'))
        self.assertIn('Do NOT describe the clothes', captions.caption_question('character', 'subject', 'woman'))
        self.assertIn('the clothes, the background', captions.caption_question('character', 'subject', 'woman', True))
        self.assertTrue(captions.suggest_trigger('Maya Lopez').isalpha())


class TrainingConfigTests(unittest.TestCase):
    def test_offload(self):
        # A 16 GB card holds neither the 21 GB int8 DiT nor it beside the 5 GB VAE: 0.53 still ran out of memory.
        self.assertEqual(training.offload_share(15.9), .8)
        self.assertEqual(training.offload_share(48), 0.)
        self.assertEqual(training.offload_share(15.9, .8), .8)
        self.assertGreater(training.offload_share(15.9, None, 768), training.offload_share(15.9))
        dataset = dict(id='d1', name='Scarlett', kind='character', trigger='scrlttv', class_word='woman')
        config = training.ai_toolkit_config(dataset, 'C:/i', '', {}, output_dir='C:/o', vram_gb=15.9)
        self.assertIn('layer_offloading: true', config)
        self.assertIn(training.H3_TRAINING_ADAPTER, config)
        self.assertNotIn('layer_offloading', training.ai_toolkit_config(dataset, 'C:/i', '', {}, output_dir='C:/o', vram_gb=80))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.folders = Folders(self.tmp)
        self.context = Context(self.folders)
        self.store = self.context.store

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def add(self, dataset, seed, **kwargs):
        image, notes = images.prepare(picture(seed), dataset['kind'])
        return self.store.add_image(dataset['id'], image, notes=notes, **kwargs)

    def test_dataset_flow(self):
        ds = self.store.create(dict(name='Maya', kind='character', class_word='woman'))
        self.assertTrue(ds['trigger'])
        first = self.add(ds, 1)
        self.assertEqual(first['dataset']['reference'], first['item']['id'])
        self.assertIn('duplicate_of', self.add(ds, 1))
        second = self.add(ds, 5, source='generated', slot='smile')
        third = self.add(ds, 9, source='generated', slot='smile', dedupe=False)
        got = self.store.get(ds['id'])
        smile = [r for r in got['items'] if r['slot'] == 'smile']
        self.assertEqual([r['excluded'] for r in smile], [True, False])
        self.store.update_item(ds['id'], third['item']['id'], dict(caption='smiling at the camera'))
        self.store.update_item(ds['id'], first['item']['id'], dict(drift=dict(score=9, differences='none')))
        reg = self.add(ds, 13, source='reg')
        self.store.update(ds['id'], dict(settings=dict(use_regularisation=True, steps=800)))
        with self.assertRaises(ValueError):
            self.store.update(ds['id'], dict(settings=dict(steps=5)))
        with self.assertRaises(ValueError):
            self.store.update(ds['id'], dict(trigger='bad/trigger'))

        # Exports
        for layout in export.LAYOUTS:
            result = self.context.export_folder(dict(dataset=ds['id'], layout=layout))
            folder = Path(result['path'])
            txt = sorted(folder.rglob('*.txt'))
            self.assertTrue(any('smiling at the camera' in p.read_text() for p in txt))
            self.assertTrue(any(p.parent.name.startswith('1_') or p.parent.name == 'reg' for p in folder.rglob('reg_*.png')))
        self.assertTrue((Path(self.folders.get_output_directory()) / 'lora-studio' / 'Maya-ai-toolkit' / 'config.yaml').is_file())
        kohya = list((Path(self.folders.get_output_directory()) / 'lora-studio' / 'Maya-kohya' / 'img').iterdir())
        self.assertEqual(kohya[0].name, '10_%s woman' % ds['trigger'])
        data, name = export.zip_bytes(self.store, ds['id'], 'musubi')
        names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        self.assertIn('dataset.toml', names)
        self.assertIn('train.cmd', names)
        self.assertTrue(name.endswith('-musubi.zip'))

        # Plan, link, install and @Name
        plan = self.context.train_plan(dict(dataset=ds['id']))
        self.assertIn('arch: "minimax_h3"', plan['config'])
        self.assertIn('is_reg: true', plan['config'])
        source = Path(self.tmp) / 'trained.safetensors'
        source.write_bytes(b'fake')
        lora = self.context.install(ds['id'], str(source), None)
        self.assertEqual(lora['name'], 'lora-studio/%s.safetensors' % training.run_name(got))
        self.context.link(dict(dataset=ds['id'], cast_id='c123', strength=0.8))
        self.assertEqual(self.store.links()['c123']['strength'], 0.8)
        text, loras = cast_loras.extend('<Picture 1> walks', {'references': [{'path': 'x', 'cast': 'c123'}]}, None, folders=self.folders)
        self.assertEqual(len(loras), 1)
        self.assertAlmostEqual(loras[0]['strength'], 0.8)
        same = cast_loras.extend('x', {'references': [{'cast': 'c123'}]}, loras, folders=self.folders)[1]
        self.assertEqual(len(same), 1)
        self.assertEqual(cast_loras.extend('x', {'references': [{'cast': 'other'}]}, None, folders=self.folders), ('x', None))
        graph = self.context.test_graph(dict(dataset=ds['id'], strength=0.8, seed=5))
        self.assertEqual(graph['prompt']['2']['class_type'], 'FreeVideoLoRAStack')
        self.assertNotIn('loras', self.context.test_graph(dict(dataset=ds['id'], strength=0))['prompt']['1']['inputs'])

        # Delete
        self.store.delete_item(ds['id'], second['item']['id'])
        self.store.delete(ds['id'])
        self.assertEqual(self.store.list(), [])
        self.assertNotIn('c123', self.store.links())

    def test_adopt_and_stage(self):
        ds = self.store.create(dict(name='Mug', kind='object'))
        made = Path(self.folders.get_input_directory()) / 'freevideo-image'
        made.mkdir()
        (made / 'out.png').write_bytes(picture(3))
        result = self.context.adopt(dict(dataset=ds['id'], file='freevideo-image/out.png', slot='front', prompt='p', seed=1))
        staged = self.context.stage(dict(dataset=ds['id'], item=result['item']['id']))['file']
        self.assertTrue((Path(self.folders.get_input_directory()) / staged).is_file())
        for bad in ('../x.png', 'other/out.png', 'freevideo-image/../../x.png'):
            with self.assertRaises(ValueError):
                self.context.adopt(dict(dataset=ds['id'], file=bad))
        with self.assertRaises(ValueError):
            self.store.get('../../etc')


class TrainingTests(unittest.TestCase):
    def test_progress(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            log = tmp / 'log.txt'
            log.write_text('loading\nmy_lora:  10%|#  | 100/1000 [01:00<09:00, lr: 1.0e-04 loss: 3.1e-01]\r'
                           'my_lora:  12%|#  | 120/1000 [01:12<08:48, lr: 1.0e-04 loss: 2.9e-01]\r', encoding='utf-8')
            step, total, loss, tail = training.read_progress(log, 1000)
            self.assertEqual((step, total), (120, 1000))
            self.assertAlmostEqual(loss, 0.29)
            self.assertEqual(training.classify('torch.OutOfMemoryError: CUDA out of memory')[0], 'oom')
            (tmp / 'out' / 'x').mkdir(parents=True)
            for name in ('x_000000250.safetensors', 'x_000000500.safetensors', 'x.safetensors'):
                (tmp / 'out' / 'x' / name).write_bytes(b'1')
            rows = training.checkpoints(tmp / 'out')
            self.assertEqual([r['step'] for r in rows], [250, 500, None])
            self.assertTrue(rows[-1]['final'])
        finally:
            shutil.rmtree(tmp)

    def test_runner_runs_a_process(self):
        import sys
        tmp = Path(tempfile.mkdtemp())
        try:
            runner = training.Runner()
            done = []
            run = dict(id='r1', dataset='d1', log=str(tmp / 'log.txt'), output=str(tmp), steps=10)
            runner.start(run, [sys.executable, '-c', 'print("x:  50%|# | 5/10 [00:01<00:01, loss: 0.5]")'], str(tmp), None,
                         lambda result: done.append(result) or result)
            for _ in range(100):
                if done:
                    break
                import time
                time.sleep(.05)
            import time
            time.sleep(.1)
            status = runner.status()
            self.assertEqual(status['phase'], 'complete')
            self.assertEqual(status['step'], 5)
        finally:
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
