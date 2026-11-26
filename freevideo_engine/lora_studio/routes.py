"""HTTP routes of LoRA Studio under /freevideo/lora-studio, and the page itself.

The page is served from this package (not from the node's web folder) so ComfyUI does not load it as an extension.
Every write needs the page token from /api/info in the X-FreeVideo-Lora header, like the image and prompt routes.
"""
import asyncio
import json
import logging
import os
import secrets
import shutil
import sys
import uuid
from pathlib import Path

from . import captions, export, images, plans, store as store_module, training

PAGE = Path(__file__).resolve().parent / 'page'
STATIC = {'app.js': 'text/javascript', 'app.css': 'text/css', 'index.html': 'text/html'}
UPLOAD_LIMIT = 40 * 1024 * 1024
BODY_LIMIT = 1024 * 1024
PAGE_FOLDER = 'freevideo-image-ref'   # The image routes accept references from here and tidy them after the job.
log = logging.getLogger(__name__)


class Context:
    def __init__(self, folders, server=None):
        self.folders = folders
        self.server = server
        self.store = store_module.Store(folders.get_input_directory())
        self.runner = training.Runner()

    # Folders ----------------------------------------------------------------------------------------------
    def loras_root(self):
        roots = self.folders.get_folder_paths('loras')
        if not roots:
            raise ValueError('no_lora_folder')
        return roots[0]

    def output_root(self):
        return Path(self.folders.get_output_directory()) / 'lora-studio'

    def lora_listed_name(self, file_name):
        """The name ComfyUI's LoRA list uses for a file in lora-studio/ (separators differ between systems)."""
        try:
            for name in self.folders.get_filename_list('loras'):
                if name.replace('\\', '/') == 'lora-studio/' + file_name:
                    return name
        except Exception:
            pass
        return 'lora-studio/' + file_name

    # Trainer discovery ------------------------------------------------------------------------------------
    def trainer(self):
        settings = self.store.settings()
        folder = settings.get('ai_toolkit_dir') or self.guess_ai_toolkit()
        python = settings.get('ai_toolkit_python') or ''
        if folder and not python:
            for candidate in (Path(folder) / 'venv' / 'Scripts' / 'python.exe', Path(folder) / 'venv' / 'bin' / 'python',
                              Path(folder) / '.venv' / 'Scripts' / 'python.exe', Path(folder) / '.venv' / 'bin' / 'python'):
                if candidate.is_file():
                    python = str(candidate)
                    break
        ok = bool(folder) and (Path(folder) / 'run.py').is_file() and bool(python) and Path(python).is_file()
        h3 = bool(folder) and (Path(folder) / 'extensions_built_in' / 'diffusion_models' / 'minimax_h3').is_dir()
        return dict(folder=folder or '', python=python, ready=ok, h3=h3, models_dir=settings.get('models_dir', ''),
                    musubi_dir=settings.get('musubi_dir', ''))

    @staticmethod
    def guess_ai_toolkit():
        for candidate in (Path('C:/Projects/ai-toolkit'), Path.home() / 'ai-toolkit', Path('C:/ai-toolkit')):
            if (candidate / 'run.py').is_file():
                return str(candidate)
        return ''

    # GPU --------------------------------------------------------------------------------------------------
    @staticmethod
    def vram_gb():
        """The card's size, for how much of H3 the trainer keeps in system memory."""
        try:
            import torch
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if torch.cuda.is_available() else None
        except Exception:
            return None

    def gpu_busy(self):
        try:
            running, pending = self.server.prompt_queue.get_current_queue()
            return bool(running or pending)
        except Exception:
            return False

    def free_for_pictures(self, body=None):
        """Make room for the picture model: the video model, ComfyUI's idle models and the prompt model let go (a picture
        failed with no_memory on Jason's PC while the video model was still resident). Nothing running is touched."""
        if self.runner.status().get('phase') == 'running':
            raise ValueError('training_running')
        if self.gpu_busy():
            raise ValueError('gpu_busy')
        return dict(released=self.release_gpu(skip=('image',)))

    def release_gpu(self, skip=()):
        """Hand the card to the trainer: ComfyUI's idle models, the image worker, the prompt model, the video model."""
        released = []
        def comfy():
            from ..comfy_residency import release_native
            release_native(self.server, timeout=20)

        def image():
            from ..image_gen.service import release_idle_worker
            release_idle_worker(self.server)

        def prompt():
            from ..prompt_vlm.service import stop_prestart_for_generation
            stop_prestart_for_generation(self.server)

        def video():
            from ..resident_process import OWNER
            OWNER.release_idle('lora_training')

        steps = (('comfy', comfy), ('image', image), ('prompt', prompt), ('video', video))
        for name, step in steps:
            if name in skip:
                continue
            try:
                step()
                released.append(name)
            except Exception as error:
                log.info('[FreeVideo] LoRA Studio could not release %s before training: %s', name, error)
        return released

    # Actions ----------------------------------------------------------------------------------------------
    def info(self):
        return dict(kinds=list(plans.KINDS), plans={kind: plans.page_plan(kind) for kind in plans.KINDS},
                    defaults=store_module.DEFAULT_SETTINGS, trainer=self.trainer(), training=self.runner.status(),
                    layouts=list(export.LAYOUTS), drift_limit=captions.DRIFT_LIMIT, page_folder=PAGE_FOLDER)

    def stage(self, body):
        """Copy a dataset picture to where the image routes take references from; they remove it after the job."""
        dataset = self.store.get(body.get('dataset'))
        item = next((row for row in dataset['items'] if row['id'] == body.get('item')), None)
        if item is None:
            raise ValueError('invalid_item')
        folder = Path(self.folders.get_input_directory()) / PAGE_FOLDER
        folder.mkdir(exist_ok=True)
        name = '%s.png' % uuid.uuid4()
        shutil.copyfile(self.store.item_path(dataset['id'], item), folder / name)
        return dict(file='%s/%s' % (PAGE_FOLDER, name))

    def _input_file(self, name, folders=('freevideo-image', PAGE_FOLDER)):
        root = Path(self.folders.get_input_directory()).resolve()
        if not isinstance(name, str) or '\\' in name or '..' in name.split('/') or ':' in name:
            raise ValueError('invalid_media')
        folder, _, leaf = name.rpartition('/')
        if folder not in folders or not leaf:
            raise ValueError('invalid_media')
        path = (root / name).resolve()
        if path.parent != (root / folder).resolve() or not path.is_file():
            raise ValueError('invalid_media')
        return path

    def adopt(self, body):
        """Take a picture the image model made (freevideo-image/<name>) into the dataset, in a slot."""
        path = self._input_file(body.get('file'))
        image = images.fit(images.open_image(str(path)))
        source = body.get('source', 'generated')
        return self.store.add_image(body.get('dataset'), image, source=source, slot=body.get('slot') or None,
                                    prompt=str(body.get('prompt') or '')[:4000], seed=body.get('seed'),
                                    model=str(body.get('model') or '')[:80], dedupe=False)

    def cutout(self, body):
        dataset = self.store.get(body.get('dataset'))
        item = next((row for row in dataset['items'] if row['id'] == body.get('item')), None)
        if item is None:
            raise ValueError('invalid_item')
        mask = images.open_image(str(self._input_file(body.get('mask'), folders=('freevideo-image',))))
        picture = images.open_image(str(self.store.item_path(dataset['id'], item)))
        return self.store.replace_image(dataset['id'], item['id'], images.cutout(picture, mask), 'background_removed')

    def slot_prompt(self, body):
        dataset = self.store.get(body.get('dataset'))
        kind = dataset['kind']
        if body.get('slot') == 'anchor':
            prompt, aspect = plans.anchor_prompt(kind, dataset.get('description') or dataset['name'])
            return dict(prompt=prompt, width=plans.size(aspect)[0], height=plans.size(aspect)[1], references=False)
        slot = plans.find(kind, body.get('slot'))
        vary = bool((dataset.get('settings') or {}).get('vary_clothes'))
        prompt = plans.slot_prompt(kind, slot, dataset.get('description', ''), keep_clothes=not vary)
        return dict(prompt=prompt, width=plans.size(slot['aspect'])[0], height=plans.size(slot['aspect'])[1], references=True)

    def reg_prompts(self, body):
        dataset = self.store.get(body.get('dataset'))
        count = body.get('count', 8)
        if type(count) is not int or not 1 <= count <= 100:
            raise ValueError('invalid_request')
        return dict(prompts=plans.regularisation_prompts(dataset['kind'], dataset.get('class_word'), count))

    def questions(self, body):
        dataset = self.store.get(body.get('dataset'))
        vary = bool((dataset.get('settings') or {}).get('vary_clothes'))
        return dict(caption_system=captions.SYSTEM,
                    caption=captions.caption_question(dataset['kind'], dataset.get('mode'), dataset.get('class_word'), vary),
                    drift_system=captions.DRIFT_SYSTEM, drift=captions.drift_question(dataset['kind'], vary))

    def finish_caption(self, body):
        """Store a caption from the prompt model, cleaned, unless the user has already written their own."""
        dataset = self.store.get(body.get('dataset'))
        item = next((row for row in dataset['items'] if row['id'] == body.get('item')), None)
        if item is None:
            raise ValueError('invalid_item')
        if item.get('caption_edited') and not body.get('overwrite'):
            return dataset
        return self.store.update_item(dataset['id'], item['id'], dict(caption=captions.clean(body.get('text')), edited=False))

    def finish_drift(self, body):
        return self.store.update_item(body.get('dataset'), body.get('item'), dict(drift=captions.parse_drift(body.get('answer') or {})))

    def export_folder(self, body):
        dataset = self.store.get(body.get('dataset'))
        layout = body.get('layout')
        destination = self.output_root() / ('%s-%s' % (export.safe_name(dataset['name']), layout))
        return export.write_folder(self.store, dataset['id'], layout, destination)

    def train_plan(self, body):
        dataset = self.store.get(body.get('dataset'))
        settings = dataset.get('settings') or {}
        trainer = self.trainer()
        _, kept, reg = self.store.training_items(dataset['id'])
        warnings = []
        if len(kept) < 10:
            warnings.append('Only %d pictures: 15 to 40 varied pictures train a far better LoRA.' % len(kept))
        if sum(1 for row in kept if not row.get('caption')) > 0:
            warnings.append('%d pictures have no caption yet; they train with the trigger word alone.' % sum(1 for row in kept if not row.get('caption')))
        drifted = [row for row in kept if (row.get('drift') or {}).get('score') is not None and row['drift']['score'] < captions.DRIFT_LIMIT]
        if drifted:
            warnings.append('%d pictures failed the likeness check; exclude or regenerate them first.' % len(drifted))
        if any(row.get('quality', {}).get('blurry') for row in kept):
            warnings.append('Some pictures look blurry; blur in the set becomes blur in the LoRA.')
        if settings.get('use_regularisation') and not reg:
            warnings.append('Regularisation is on but there are no regularisation pictures; it will be skipped.')
        try:
            free = shutil.disk_usage(trainer['models_dir'] or trainer['folder'] or self.folders.get_output_directory()).free
            if free < 50 * 1024 ** 3:
                warnings.append('Only %.0f GB free; the H3 model files for training take about 40 GB on the first run.' % (free / 1024 ** 3))
        except OSError:
            pass
        steps = int(settings.get('steps', 1500))
        # The published 12 GB run took ~6.5 h for 1000 steps at 512 px; a 16 GB card swaps less.
        # Measured on Jason's RTX 5080 (16 GB, offload 0.8): 1.41 s a step at 512 px, about 10 minutes of loading and
        # caching before the first step. Steps scale with the pixels.
        hours = (600 + steps * 1.45 * (int(settings.get('resolution', 512)) / 512) ** 2) / 3600
        destination = self.output_root() / ('%s-ai-toolkit' % export.safe_name(dataset['name']))
        config = training.ai_toolkit_config(
            dataset, str(destination / 'images'), str(destination / 'reg') if settings.get('use_regularisation') and reg else '',
            settings, output_dir=str(destination / 'output'), models_dir=trainer['models_dir'], vram_gb=self.vram_gb())
        vram = self.vram_gb()
        offload = training.offload_share(vram, settings.get('offload'))
        return dict(trainer=trainer, warnings=warnings, pictures=len(kept), offload=offload, vram_gb=round(vram or 0, 1), reg=len(reg), estimate_hours=round(hours, 2),
                    config=config, gpu_busy=self.gpu_busy(), training=self.runner.status())

    def train_start(self, body):
        dataset = self.store.get(body.get('dataset'))
        trainer = self.trainer()
        if not trainer['ready']:
            raise ValueError('trainer_missing')
        if self.gpu_busy():
            raise ValueError('gpu_busy')
        settings = dataset.get('settings') or {}
        destination = self.output_root() / ('%s-ai-toolkit' % export.safe_name(dataset['name']))
        exported = export.write_folder(self.store, dataset['id'], 'ai-toolkit', destination)
        run_id = store_module.new_id('d').replace('d', 'r', 1)
        run_dir = self.store.root / dataset['id'] / 'runs' / run_id
        run_dir.mkdir(parents=True)
        output = destination / 'output'
        output.mkdir(exist_ok=True)
        config = run_dir / 'config.yaml'
        use_reg = (destination / 'reg').is_dir()
        config.write_text(training.ai_toolkit_config(
            dataset, str(destination / 'images'), str(destination / 'reg') if use_reg else '', settings,
            output_dir=str(output), models_dir=trainer['models_dir'], vram_gb=self.vram_gb()), encoding='utf-8')
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        for key in list(env):
            # The trainer has its own Python: ComfyUI's paths must not leak into it.
            if key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
                env.pop(key)
        if trainer['models_dir']:
            env['MODELS_PATH'] = trainer['models_dir']
        released = self.release_gpu()
        run = dict(id=run_id, dataset=dataset['id'], log=str(run_dir / 'log.txt'), config=str(config), output=str(output),
                   steps=int(settings.get('steps', 1500)), trainer='ai-toolkit', exported=exported['files'],
                   name=training.run_name(dataset), released=released)
        self.store.record(dataset['id'], 'runs', dict(id=run_id, started_at=store_module.now(), phase='running',
                                                     steps=run['steps'], settings=settings))
        return self.runner.start(run, [trainer['python'], 'run.py', str(config)], trainer['folder'], env, self.finished)

    def finished(self, result):
        """When a run ends: record it, and install the final LoRA so the creation panel and @Name can use it."""
        dataset_id = result.get('dataset')
        entry = dict(id=result.get('id'), phase=result.get('phase'), ended_at=store_module.now(),
                     error=result.get('error'), steps=result.get('steps'))
        if result.get('phase') == 'complete':
            rows = training.checkpoints(result['output'])
            final = next((row for row in reversed(rows) if row['final']), rows[-1] if rows else None)
            if final is None:
                result.update(phase='failed', error='no_output', message='The trainer ended without writing a LoRA file.')
                entry.update(phase='failed', error='no_output')
            else:
                lora = self.install(dataset_id, final['path'], None)
                result['lora'] = lora
                entry['lora'] = lora['name']
        try:
            self.store.record(dataset_id, 'runs', entry)
        except ValueError:
            pass
        return result

    def install(self, dataset_id, source, step):
        dataset = self.store.get(dataset_id)
        file_name = '%s%s.safetensors' % (training.run_name(dataset), '' if step is None else '_step%d' % step)
        listed, path = training.install_lora(source, self.loras_root(), file_name)
        name = self.lora_listed_name(file_name)
        try:
            from ..lora_triggers import set_trigger
            lead = dataset['trigger'] if dataset['kind'] == 'style' else ('%s %s' % (dataset['trigger'], dataset.get('class_word') or '')).strip()
            set_trigger(name, lead, folders=self.folders)
        except Exception as error:
            log.info('[FreeVideo] LoRA Studio could not save the trigger words: %s', error)
        lora = dict(name=name, file=path, step=step, installed_at=store_module.now(),
                    strength=(dataset.get('lora') or {}).get('strength', 1.0),
                    cast_id=(dataset.get('lora') or {}).get('cast_id'))
        self.store.set_lora(dataset_id, lora)
        if lora.get('cast_id'):
            self.store.link(lora['cast_id'], dict(dataset=dataset_id, lora=name, strength=lora['strength'], trigger=dataset['trigger']))
        return lora

    def promote(self, body):
        """Use an earlier checkpoint of the last run (when the final one overfits)."""
        dataset = self.store.get(body.get('dataset'))
        status = self.runner.status()
        output = status.get('output') if status.get('dataset') == dataset['id'] else None
        if not output:
            destination = self.output_root() / ('%s-ai-toolkit' % export.safe_name(dataset['name'])) / 'output'
            output = str(destination)
        row = next((c for c in training.checkpoints(output) if c['name'] == body.get('checkpoint')), None)
        if row is None:
            raise ValueError('unknown_checkpoint')
        return dict(lora=self.install(dataset['id'], row['path'], row['step']), dataset=self.store.get(dataset['id']))

    def checkpoint_list(self, body):
        dataset = self.store.get(body.get('dataset'))
        output = self.output_root() / ('%s-ai-toolkit' % export.safe_name(dataset['name'])) / 'output'
        return dict(checkpoints=[dict(name=c['name'], step=c['step'], final=c['final'], bytes=c['bytes'])
                                 for c in training.checkpoints(output)] if output.is_dir() else [])

    def link(self, body):
        """Tie the dataset's LoRA to a cast card (or untie it with cast_id null)."""
        dataset = self.store.get(body.get('dataset'))
        lora = dict(dataset.get('lora') or {})
        cast_id = body.get('cast_id')
        if 'strength' in body:
            strength = body['strength']
            if type(strength) not in (int, float) or not 0 <= strength <= 2:
                raise ValueError('invalid_request')
            lora['strength'] = float(strength)
        if lora.get('cast_id') and lora['cast_id'] != cast_id:
            self.store.link(lora['cast_id'], None)
        if cast_id:
            if not lora.get('name'):
                raise ValueError('no_lora')
            self.store.link(cast_id, dict(dataset=dataset['id'], lora=lora['name'], strength=lora.get('strength', 1.0),
                                          trigger=dataset['trigger']))
        lora['cast_id'] = cast_id or None
        return self.store.set_lora(dataset['id'], lora)

    def set_lora_file(self, body):
        """Point the dataset at a LoRA trained elsewhere (a file already in ComfyUI's LoRA list)."""
        dataset = self.store.get(body.get('dataset'))
        name = body.get('name')
        if not isinstance(name, str) or name not in self.folders.get_filename_list('loras'):
            raise ValueError('lora_not_found')
        lora = dict(dataset.get('lora') or {}, name=name, file=None, step=None, installed_at=store_module.now())
        lora.setdefault('strength', 1.0)
        if lora.get('cast_id'):
            self.store.link(lora['cast_id'], dict(dataset=dataset['id'], lora=name, strength=lora['strength'], trigger=dataset['trigger']))
        return self.store.set_lora(dataset['id'], lora)

    def test_graph(self, body):
        """A ComfyUI prompt (API format) for one short test video with the dataset's LoRA at one strength."""
        dataset = self.store.get(body.get('dataset'))
        lora = dataset.get('lora') or {}
        if not lora.get('name'):
            raise ValueError('no_lora')
        strength = body.get('strength', 1.0)
        seed = body.get('seed', 1234)
        seconds = body.get('seconds', 3.0)
        prompt = body.get('prompt') or (dataset.get('settings') or {}).get('test_prompt') or training.default_test_prompt(dataset)
        if (type(strength) not in (int, float) or not 0 <= strength <= 2 or type(seed) is not int or not 0 <= seed < 2 ** 53
                or type(seconds) not in (int, float) or not 1.625 <= seconds <= 10 or not isinstance(prompt, str) or len(prompt) > 4000):
            raise ValueError('invalid_request')
        portrait = dataset['kind'] == 'character' and body.get('portrait', False)
        width, height = (480, 832) if portrait else (832, 480)
        adapters = [dict(name=lora['name'], strength=float(strength), enabled=True)] if strength else []
        graph = {'1': dict(class_type='FreeVideoGenerate', inputs=dict(text=prompt, width=width, height=height,
                                                                      seconds=float(seconds), seed=seed))}
        if adapters:
            graph['2'] = dict(class_type='FreeVideoLoRAStack', inputs=dict(adapters=json.dumps(adapters)))
            graph['1']['inputs']['loras'] = ['2', 0]
        return dict(prompt=graph, text=prompt, strength=strength, seed=seed, width=width, height=height)

    def test_record(self, body):
        dataset = self.store.get(body.get('dataset'))
        test = body.get('test')
        if not isinstance(test, dict) or len(json.dumps(test)) > 20000:
            raise ValueError('invalid_request')
        test = dict(test, id=test.get('id') or store_module.new_id('d').replace('d', 't', 1), at=store_module.now())
        return self.store.record(dataset['id'], 'tests', test, limit=10)


def register():
    from aiohttp import web
    from server import PromptServer
    import folder_paths
    server = PromptServer.instance
    if server is None or getattr(server, '_freevideo_lora_studio', None) is not None:
        return
    context = Context(folder_paths, server)
    server._freevideo_lora_studio = context
    token = secrets.token_urlsafe(24)
    no_store = {'Cache-Control': 'no-store'}

    def page_file(name):
        return web.Response(body=(PAGE / name).read_bytes(), content_type=STATIC[name], headers=no_store)

    @server.routes.get('/freevideo/lora-studio')
    async def page_redirect(request):
        raise web.HTTPFound('/freevideo/lora-studio/')

    @server.routes.get('/freevideo/lora-studio/')
    async def page(request):
        return page_file('index.html')

    @server.routes.get('/freevideo/lora-studio/static/{name}')
    async def static(request):
        name = request.match_info['name']
        if name not in STATIC or name == 'index.html':
            raise web.HTTPNotFound()
        return page_file(name)

    @server.routes.get('/freevideo/lora-studio/api/info')
    async def info(request):
        try:
            return web.json_response(dict(await asyncio.to_thread(context.info), token=token), headers=no_store)
        except Exception:
            log.exception('[FreeVideo] LoRA Studio info failed')
            return web.json_response(dict(error='unavailable'), status=500)

    @server.routes.get('/freevideo/lora-studio/api/links')
    async def links(request):
        # Read by the creation panel to load a card's LoRA when the prompt mentions it; holds no secrets.
        return web.json_response(await asyncio.to_thread(context.store.links), headers=no_store)

    def checked(request, value=None):
        given = request.headers.get('X-FreeVideo-Lora', '') or (value or '')
        if not secrets.compare_digest(given.encode('utf-8'), token.encode('utf-8')):
            raise PermissionError('reload')

    def failure(error):
        if isinstance(error, PermissionError):
            return web.json_response(dict(error='reload'), status=403)
        code = str(error) if isinstance(error, ValueError) else 'failed'
        if not isinstance(error, ValueError):
            log.exception('[FreeVideo] LoRA Studio action failed')
        return web.json_response(dict(error=code), status=400 if isinstance(error, ValueError) else 500)

    @server.routes.get('/freevideo/lora-studio/api/zip')
    async def download(request):
        try:
            checked(request, request.query.get('token'))
            data, name = await asyncio.to_thread(export.zip_bytes, context.store, request.query.get('dataset'),
                                                 request.query.get('layout'))
            return web.Response(body=data, content_type='application/zip',
                                headers={'Content-Disposition': 'attachment; filename="%s"' % name, **no_store})
        except Exception as error:
            return failure(error)

    @server.routes.post('/freevideo/lora-studio/api/upload')
    async def upload(request):
        try:
            checked(request)
            if request.content_length and request.content_length > UPLOAD_LIMIT:
                raise ValueError('too_large')
            data = await read_all(request, UPLOAD_LIMIT)
            dataset_id = request.query.get('dataset')
            source = request.query.get('source', 'import')
            crop = request.query.get('crop', '1') == '1'

            def work():
                dataset = context.store.get(dataset_id)
                faces = None
                if crop and dataset['kind'] == 'character' and source == 'import':
                    faces = find_faces(data)
                image, notes = images.prepare(data, dataset['kind'] if source == 'import' else 'style', faces, crop)
                return context.store.add_image(dataset_id, image, source=source, notes=notes,
                                               dedupe=request.query.get('dedupe', '1') == '1')
            return web.json_response(await asyncio.to_thread(work), headers=no_store)
        except Exception as error:
            return failure(error)

    actions = {
        'list': lambda body: dict(datasets=context.store.list()),
        'get': lambda body: context.store.get(body.get('dataset')),
        'create': lambda body: context.store.create(body),
        'update': lambda body: context.store.update(body.get('dataset'), {k: v for k, v in body.items() if k != 'dataset'}),
        'delete': lambda body: context.store.delete(body.get('dataset')),
        'item-update': lambda body: context.store.update_item(body.get('dataset'), body.get('item'),
                                                              {k: v for k, v in body.items() if k not in ('dataset', 'item')}),
        'item-delete': lambda body: context.store.delete_item(body.get('dataset'), body.get('item')),
        'dedupe': lambda body: context.store.dedupe(body.get('dataset')),
        'adopt': context.adopt,
        'stage': context.stage,
        'cutout': context.cutout,
        'slot-prompt': context.slot_prompt,
        'reg-prompts': context.reg_prompts,
        'questions': context.questions,
        'caption': context.finish_caption,
        'drift': context.finish_drift,
        'export': context.export_folder,
        'train-plan': context.train_plan,
        'train-start': context.train_start,
        'train-status': lambda body: context.runner.status(),
        'free-gpu': context.free_for_pictures,
        'train-cancel': lambda body: context.runner.cancel(),
        'checkpoints': context.checkpoint_list,
        'promote': context.promote,
        'link': context.link,
        'use-lora': context.set_lora_file,
        'settings': lambda body: dict(settings=context.store.save_settings(body), trainer=context.trainer()),
        'test-graph': context.test_graph,
        'test-record': context.test_record,
    }

    @server.routes.post('/freevideo/lora-studio/api/{action}')
    async def action(request):
        try:
            checked(request)
            handler = actions.get(request.match_info['action'])
            if handler is None:
                return web.json_response(dict(error='unknown_action'), status=404)
            if request.content_length and request.content_length > BODY_LIMIT:
                raise ValueError('invalid_request')
            raw = await read_all(request, BODY_LIMIT)
            body = json.loads(raw or b'{}')
            if not isinstance(body, dict):
                raise ValueError('invalid_request')
            return web.json_response(await asyncio.to_thread(handler, body), headers=no_store)
        except json.JSONDecodeError:
            return web.json_response(dict(error='invalid_request'), status=400)
        except Exception as error:
            return failure(error)


async def read_all(request, limit):
    """The whole request body (a single read returns only what has arrived so far), refused past `limit`."""
    body = bytearray()
    while True:
        chunk = await request.content.read(min(1024 * 1024, limit + 1 - len(body)))
        if not chunk:
            return bytes(body)
        body.extend(chunk)
        if len(body) > limit:
            raise ValueError('too_large')


def find_faces(data):
    """Face boxes of an uploaded photo from the engine's own CPU face finder, or None when it is not available."""
    import tempfile
    try:
        from ..image_gen.face_box import detect
    except Exception:
        return None
    handle, path = tempfile.mkstemp(suffix='.img')
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data)
        return detect(path).get('faces')
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
