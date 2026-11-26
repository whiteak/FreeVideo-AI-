# LoRA Studio (fork addition)

Make a LoRA of a **character**, an **object** or a **style**, train it on this PC as a MiniMax H3 video LoRA, and
load it by writing `@Name` in a prompt. Open it from the **LoRA** entry in FreeVideo's top bar, or go to
`http://127.0.0.1:8188/freevideo/lora-studio/`.

## The steps

1. **Pictures.** Drop in your own photos. Each one is straightened, black bars are trimmed, a person who is small in
   a big photo is cropped to (using the engine's face finder), and near-duplicates are skipped. For objects,
   *Remove background* cuts the object out with the SAM 2 selection model and puts it on plain grey.
   *Add face close-ups* (characters) cuts head-and-shoulders and half-length crops from the pictures, so the face
   is big in part of the set; without them a set of full-body shots teaches the outfit better than the face.
   Pick the clearest picture as the **reference**.
   **Your own pictures come first.** With 8 or more of your own pictures (a character pack), training uses only
   those and close-ups cut from them, and generated coverage pictures are left out: real pictures teach the face
   better. Change it under Train → *Train on*.
2. **Coverage.** A grid of slots for what a good set covers: angles, close-ups, expressions, poses, places and light
   for characters (29 slots); sides, details, places, in-use and light for objects (21); varied subjects for a style
   (18). *Fill empty slots* makes each one with H3 Image as an edit of the reference, so the subject stays the
   same. *Regenerate* makes a new picture in one slot (the old one is kept, excluded). No photo? Write a description
   and make the reference from it. Optional **regularisation** pictures (generic pictures of the class) are made with
   MOSS-Image.
3. **Captions & check.** The installed prompt model (Qwen3-VL) writes a caption for every picture. In subject mode
   the caption describes everything *except* the subject (pose, angle, place, light), so the trigger word learns
   the subject. A character's outfit counts as part of the character and is left out unless
   *Clothes change between pictures* is ticked. In style mode it describes the content and never the style. Every caption is editable.
   *Check likeness* compares each picture with the reference and scores it 0-10; drifted ones (below 6) are
   flagged for regenerating or excluding. A picture where the face can't be seen (from behind) scores 7 at most.
4. **Train.** Trains with [ai-toolkit](https://github.com/ostris/ai-toolkit) (arch `minimax_h3`) in its own Python:
   pruned int8 ConvRot H3 transformer, nvfp4 text encoder, `low_vram`, ostris's H3 training adapter (keeps the
   distilled model intact), single-frame image training at 512 px, rank 16, 1500 steps by default, with a checkpoint saved
   every 125 steps (the best one is often well before the last; test a few). The 21 GB
   transformer doesn't fit a 16 GB card, so part of it waits in system memory and is streamed in layer by layer
   (*Offload to RAM*, worked out from the card's size; about 80% on 16 GB, since the 5 GB video VAE also sits on the card). On an RTX 5080 (16 GB) the default 1500 steps took about 35 minutes (1.4 s a
   step) plus about 10 minutes of loading. The first run downloads about 40 GB of H3 files from
   [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (accept the licence there first).
   Progress, loss and the trainer log show on the page; each saved checkpoint can be used instead of the final one.
   The finished LoRA is copied to `models/loras/lora-studio/` and its trigger words are saved, so the creation panel
   adds them for you. FreeVideo can't make videos while training runs.
   **Export instead** writes kohya, ai-toolkit or Musubi Tuner layouts (pictures + `.txt` captions, plus a config
   for ai-toolkit and a `dataset.toml` / `train.cmd` for Musubi) to `output/lora-studio/`, or as a zip.
5. **Use & test.** Link the LoRA to a cast card (or make a new card from the set). When that card is mentioned with
   `@` in a prompt, its LoRA is added to the request at the strength you set and its trigger words go in front of the
   prompt; the panel's own LoRA list is not changed. The **strength test grid** makes short videos with the same
   prompt and seed without the LoRA and at several strengths.

## Installing the trainer

```
cd C:\Projects\FreeVideo-AI
powershell -ExecutionPolicy Bypass -File scripts\install-lora-trainer.ps1
```

Then accept the licence at https://huggingface.co/Comfy-Org/MiniMax-H3 and log in with a Read token from
https://huggingface.co/settings/tokens: `C:\Projects\ai-toolkit\venv\Scripts\hf.exe auth login`.

It clones ai-toolkit to `C:\Projects\ai-toolkit`, makes its own `venv`, and installs PyTorch for CUDA 13 and the
requirements. LoRA Studio finds that folder by itself; another folder can be set under Train → Trainer setup, with
an optional models folder (`MODELS_PATH`) for the 40 GB of training files.

## Where things are

| What | Where |
|---|---|
| Datasets (pictures, captions, settings, runs) | `ComfyUI/input/lora-studio/<id>/` |
| Cast card → LoRA links | `ComfyUI/input/lora-studio/links.json` |
| Exports and the training copy of the set | `ComfyUI/output/lora-studio/` |
| Finished LoRAs | `ComfyUI/models/loras/lora-studio/` |
| Code | `freevideo_engine/lora_studio/`, `web/lora_studio_link.js`, the `@Name` hook in `FreeVideoGenerate.execute` |

Tests: `python -m unittest freevideo_engine.lora_studio.tests.test_lora_studio` (no GPU or models needed).

## Limits

- H3 Image and MOSS-Image in FreeVideo don't take LoRAs, so the finished LoRA is a **video** LoRA; test it with the
  test grid or the creation panel.
- Pictures made for a set also appear in the Images view's **Creations**, like every picture FreeVideo makes; the set
  keeps its own copies under `input/lora-studio/`, so you can delete them from Creations without losing anything.
- The likeness check is the prompt model's judgement, not a face-recognition score: use it to find outliers.
- Musubi's H3 recipe is new; check `train.cmd` against `docs/minimax_h3.md` in your Musubi version.
