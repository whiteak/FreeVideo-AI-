"""Train a MiniMax H3 LoRA with an external trainer and bring the result into ComfyUI's LoRA folder.

The trainer is ai-toolkit (ostris/ai-toolkit, arch ``minimax_h3``) by default: it trains H3 on still pictures as
single frames from the Comfy-Org repack (pruned int8 ConvRot transformer, nvfp4 Qwen3-VL text encoder) with
``low_vram``, which is what fits a 16 GB card. A published 12 GB run used the same settings (512 px, rank 16,
1000 steps, about 6-7 hours for 32 pictures). Musubi Tuner files are written too, for people who prefer it.

FreeVideo's own LoRA loader reads the key names both trainers write (``blocks.N.attn.qkv_proj`` and friends, with
or without a ``diffusion_model.`` prefix), so the finished file works in the creation panel as it is.

The trainer runs as its own process with its own Python; this process only starts it, reads its log and stops it.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

H3_REPO = 'Comfy-Org/MiniMax-H3'
PROGRESS = re.compile(r'(\d+)/(\d+)\s*\[')
LOSS = re.compile(r'loss[:=]\s*([0-9.]+(?:e[-+]?\d+)?)', re.IGNORECASE)
ERROR_HINTS = (
    ('CUDA out of memory', 'oom', 'The graphics card ran out of memory. Close other GPU programs, lower the resolution to 512 or the rank to 8, and try again.'),
    ('OutOfMemoryError', 'oom', 'The graphics card ran out of memory. Close other GPU programs, lower the resolution to 512 or the rank to 8, and try again.'),
    ('No module named', 'missing_module', 'The trainer\'s Python is missing a package. Run the trainer installer again (scripts\\install-lora-trainer.ps1).'),
    ('401 Client Error', 'hf_auth', 'Hugging Face refused the model download. Accept the MiniMax H3 licence on huggingface.co and log in with "huggingface-cli login" in the trainer\'s Python.'),
    ('gated', 'hf_auth', 'Hugging Face refused the model download. Accept the MiniMax H3 licence on huggingface.co and log in with "huggingface-cli login" in the trainer\'s Python.'),
    ('No space left', 'disk_full', 'The disk is full. The trainer needs about 45 GB for the H3 model files plus room for checkpoints.'),
    ('not enough space', 'disk_full', 'The disk is full. The trainer needs about 45 GB for the H3 model files plus room for checkpoints.'),
    ('no images found', 'no_images', 'The trainer found no pictures. Export again and check the dataset has pictures that are not excluded.'),
)


def _q(text):
    """A YAML double-quoted string."""
    return json.dumps(str(text), ensure_ascii=False)


def run_name(dataset):
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', dataset.get('name') or 'lora').strip('_')[:40] or 'lora'
    return '%s_%s' % (name, dataset.get('trigger', 'x').replace(' ', '_'))


def ai_toolkit_config(dataset, images_dir, reg_dir, settings, *, output_dir, models_dir=''):
    s = settings
    name = run_name(dataset)
    resolution = int(s.get('resolution', 512))
    resolutions = sorted({resolution, *([768] if resolution > 768 else [])})
    reg = ''
    if reg_dir:
        reg = ('        - folder_path: %s\n'
               '          caption_ext: "txt"\n'
               '          is_reg: true\n'
               '          num_frames: 1\n'
               '          auto_frame_count: false\n'
               '          resolution: %s\n') % (_q(reg_dir), json.dumps(resolutions))
    sample_prompt = (s.get('test_prompt') or '').strip() or default_test_prompt(dataset)
    return f"""# Written by FreeVideo LoRA Studio for ai-toolkit (ostris/ai-toolkit), arch minimax_h3.
# Run from the ai-toolkit folder:  python run.py "<this file>"
# Models come from {H3_REPO} on Hugging Face (accept the MiniMax H3 licence first); about 40 GB on first run,
# kept in MODELS_PATH{(' = ' + models_dir) if models_dir else ''}.
---
job: extension
config:
  name: {_q(name)}
  process:
    - type: "sd_trainer"
      training_folder: {_q(output_dir)}
      device: cuda:0
      network:
        type: "lora"
        linear: {int(s.get('rank', 16))}
        linear_alpha: {int(s.get('alpha', 16))}
      save:
        dtype: float16
        save_every: {int(s.get('save_every', 250))}
        max_step_saves_to_keep: 8
      datasets:
        - folder_path: {_q(images_dir)}
          caption_ext: "txt"
          caption_dropout_rate: {float(s.get('caption_dropout', .05))}
          num_frames: 1
          auto_frame_count: false
          resolution: {json.dumps(resolutions)}
{reg}      train:
        batch_size: 1
        steps: {int(s.get('steps', 1500))}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: {float(s.get('learning_rate', 1e-4))}
        optimizer_params:
          weight_decay: 0.0001
        dtype: bf16
        cache_text_embeddings: true
        audio_loss_multiplier: 0.0
        disable_sampling: true
      model:
        name_or_path: {_q(H3_REPO)}
        arch: "minimax_h3"
        quantize: true
        qtype: "convrot8"
        quantize_te: true
        qtype_te: "nvfp4"
        low_vram: true
        model_kwargs:
          partition: "fl2va_pruned"
      sample:
        sampler: "flowmatch"
        sample_every: 1000000
        width: 768
        height: 768
        num_frames: 1
        prompts:
          - {_q(sample_prompt)}
meta:
  name: "[name]"
  version: "1.0"
  made_with: "FreeVideo LoRA Studio"
"""


def musubi_files(dataset, images_dir, reg_dir, settings, *, cache_dir, output_dir):
    s = settings
    name = run_name(dataset)
    resolution = int(s.get('resolution', 512))
    toml = ('# Written by FreeVideo LoRA Studio for Musubi Tuner (kohya-ss/musubi-tuner), MiniMax H3 one-frame image training.\n'
            '[general]\nresolution = [%d, %d]\ncaption_extension = ".txt"\nbatch_size = 1\nenable_bucket = true\n'
            'bucket_no_upscale = false\n\n[[datasets]]\nimage_directory = %s\ncache_directory = %s\nnum_repeats = %d\n'
            % (resolution, resolution, _q(images_dir), _q(cache_dir), int(s.get('repeats', 10))))
    if reg_dir:
        toml += '\n[[datasets]]\nimage_directory = %s\ncache_directory = %s\nnum_repeats = 1\n' % (
            _q(reg_dir), _q(str(Path(cache_dir) / 'reg')))
    epochs = max(1, round(int(s.get('steps', 1500)) / max(1, int(s.get('repeats', 10)) * max(1, dataset_count(images_dir)))))
    cmd = f"""@echo off
rem Written by FreeVideo LoRA Studio. Run from the musubi-tuner folder with its venv active.
rem Check the flags against docs\\minimax_h3.md in your Musubi version: the H3 recipe is new and still moving.
rem Weights: Comfy-Org/MiniMax-H3 (diffusion_models, text_encoders, vae). Set the paths below first.
set DIT=C:\\models\\diffusion_models\\minimax_h3_fl2va_pruned_int8_convrot.safetensors
set TE=C:\\models\\text_encoders\\qwen3vl_32b_minimax_h3_bf16.safetensors
set VVAE=C:\\models\\vae\\minimax_h3_video_vae_fp16.safetensors
set AVAE=C:\\models\\vae\\minimax_h3_audio_vae_fp32.safetensors
set CFG=%~dp0dataset.toml

python src\\musubi_tuner\\minimax_h3_cache_latents.py --dataset_config "%CFG%" --task t2va --one_frame --video_vae "%VVAE%" --audio_vae "%AVAE%" --skip_existing || exit /b 1
python src\\musubi_tuner\\minimax_h3_cache_text_encoder_outputs.py --dataset_config "%CFG%" --task t2va --text_encoder "%TE%" --text_cache_dtype bf16 --uncond_output "%~dp0uncond.safetensors" --skip_existing || exit /b 1
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src\\musubi_tuner\\minimax_h3_train_network.py ^
  --dataset_config "%CFG%" --task t2va --one_frame --video_only --dit "%DIT%" ^
  --network_module networks.lora_minimax_h3 --network_dim {int(s.get('rank', 16))} --network_alpha {int(s.get('alpha', 16))} ^
  --sdpa --mixed_precision bf16 --gradient_checkpointing --blocks_to_swap 40 ^
  --optimizer_type adamw8bit --learning_rate {float(s.get('learning_rate', 1e-4))} --max_train_epochs {epochs} --save_every_n_epochs 1 ^
  --h3_guidance_loss_scale 4.0 --h3_guidance_loss_sigma_min 0.15 --h3_guidance_loss_uncond_cache "%~dp0uncond.safetensors" ^
  --output_dir "{output_dir}" --output_name "{name}"
"""
    return {'dataset.toml': toml, 'train.cmd': cmd}


def dataset_count(images_dir):
    try:
        return sum(1 for p in Path(images_dir).iterdir() if p.suffix.lower() == '.png')
    except OSError:
        return 20


def default_test_prompt(dataset):
    trigger, cls = dataset.get('trigger', ''), dataset.get('class_word') or ''
    lead = ('%s %s' % (trigger, cls)).strip()
    if dataset.get('kind') == 'object':
        return 'A close-up shot of %s on a wooden table, the camera slowly circles it, warm window light.' % lead
    if dataset.get('kind') == 'style':
        return '%s, a lighthouse on a rocky coast at dusk, waves crash against the rocks, the camera pushes in.' % lead
    return ('A medium shot of %s walking along a sunny city street, turning to smile at the camera, natural light.' % lead)


def layout_files(dataset, layout, destination, use_reg, settings):
    """The trainer files that go with an export (paths point into `destination`)."""
    s = dict(dataset.get('settings') or {}, **(settings or {}))
    destination = Path(destination)
    images_dir = str(destination / 'images')
    reg_dir = str(destination / 'reg') if use_reg else ''
    if layout == 'ai-toolkit':
        return {'config.yaml': ai_toolkit_config(dataset, images_dir, reg_dir, s, output_dir=str(destination / 'output'))}
    if layout == 'musubi':
        return musubi_files(dataset, images_dir, reg_dir, s, cache_dir=str(destination / 'cache'),
                            output_dir=str(destination / 'output'))
    return {}


def classify(log_tail):
    for needle, code, message in ERROR_HINTS:
        if needle.lower() in log_tail.lower():
            return code, message
    return 'trainer_failed', 'The trainer stopped with an error. The last lines of its log are below.'


def read_progress(log_path, total_steps):
    """(step, total, loss, tail text) from the end of a trainer log; tqdm's carriage returns count as line ends."""
    try:
        with open(log_path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 96 * 1024))
            text = stream.read().decode('utf-8', 'replace')
    except OSError:
        return 0, total_steps, None, ''
    lines = [line for line in re.split(r'[\r\n]+', text) if line.strip()]
    step, loss = 0, None
    for line in reversed(lines):
        match = PROGRESS.search(line)
        if match and int(match.group(2)) == total_steps:
            step = int(match.group(1))
            found = LOSS.search(line)
            loss = float(found.group(1)) if found else None
            break
    if loss is None:
        for line in reversed(lines[-200:]):
            found = LOSS.search(line)
            if found:
                loss = float(found.group(1))
                break
    return step, total_steps, loss, '\n'.join(lines[-25:])


def checkpoints(output_dir):
    """Every .safetensors the run wrote, newest last, with its step when the name carries one."""
    rows = []
    for path in Path(output_dir).rglob('*.safetensors'):
        match = re.search(r'_(\d{4,})\.safetensors$', path.name)
        rows.append(dict(path=str(path), name=path.name, step=int(match.group(1)) if match else None,
                         bytes=path.stat().st_size, final=match is None))
    rows.sort(key=lambda row: (row['final'], row['step'] or 0))
    return rows


def _kill_tree(process):
    if process.poll() is not None:
        return
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()


class Runner:
    """One training run at a time; its state survives a page reload (not a FreeVideo restart, which ends it)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.run = None  # dict(id, dataset, log, output, steps, started, trainer)
        self.finished = None
        self.on_finish = None

    def busy(self):
        return self.process is not None and self.process.poll() is None

    def start(self, run, command, cwd, env, on_finish):
        with self.lock:
            if self.busy():
                raise ValueError('training_busy')
            log = open(run['log'], 'ab', buffering=0)
            kwargs = dict(cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            if sys.platform == 'win32':
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            else:
                kwargs['start_new_session'] = True
            try:
                self.process = subprocess.Popen(command, **kwargs)
            except OSError as error:
                log.close()
                raise ValueError('trainer_not_found') from error
            self.run = dict(run, started=time.time(), pid=self.process.pid)
            self.finished = None
            self.on_finish = on_finish
            threading.Thread(target=self._wait, args=(self.process, log), name='freevideo-lora-train', daemon=True).start()
            return self.status()

    def _wait(self, process, log):
        code = process.wait()
        log.close()
        with self.lock:
            run = dict(self.run or {})
            cancelled = run.get('cancelled', False)
        result = dict(run, returncode=code, ended=time.time(),
                      phase='cancelled' if cancelled else 'complete' if code == 0 else 'failed')
        if result['phase'] == 'failed':
            _, _, _, tail = read_progress(run.get('log', ''), run.get('steps', 0))
            result['error'], result['message'] = classify(tail)
        try:
            if self.on_finish:
                result = self.on_finish(result) or result
        except Exception as error:  # The run is over either way; say why the hand-over failed.
            result.update(phase='failed', error='finish_failed', message='Training finished but the LoRA could not be copied: %s' % error)
        with self.lock:
            self.finished = result
            self.process = None

    def cancel(self):
        with self.lock:
            process = self.process
            if process is None or self.run is None:
                raise ValueError('not_training')
            self.run['cancelled'] = True
        _kill_tree(process)
        return dict(phase='cancelling')

    def status(self):
        run = self.run
        if run is None:
            return dict(phase='idle')
        if not self.busy() and self.finished is not None:
            result = dict(self.finished)
        else:
            result = dict(run, phase='running')
        step, total, loss, tail = read_progress(run['log'], run.get('steps', 0))
        result.update(step=step, total=total, loss=loss, tail=tail)
        if result['phase'] == 'running' and step:
            elapsed = time.time() - run['started']
            result['eta_seconds'] = round(elapsed / step * (total - step)) if step < total else 0
        result['checkpoints'] = [dict(name=c['name'], step=c['step'], final=c['final']) for c in checkpoints(run['output'])] \
            if run.get('output') and Path(run['output']).is_dir() else []
        return result


def install_lora(source, loras_root, file_name):
    """Copy a trained file into <loras>/lora-studio/ and return its name as ComfyUI lists it."""
    target_dir = Path(loras_root) / 'lora-studio'
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name
    temporary = target.with_name(target.name + '.tmp')
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return 'lora-studio/' + file_name, str(target)
