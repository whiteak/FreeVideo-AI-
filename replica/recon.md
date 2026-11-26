# Recon: FreeVideo

Date: 2026-11-26. Mapped from the public, Apache-2.0 repository (your fork `whiteak/FreeVideo-AI-`, upstream commit `e6a8e8f`, release 0.6.13), its README and docs, and its bundled tutorial (44 chapters) and feature tour (19 cards). The app was not run: it needs an NVIDIA GPU (8 GB+ VRAM) or an Apple silicon Mac, so states that only show at runtime are marked *not seen*.

## 1. Scope

- **App:** FreeVideo, a local AI video generator. It runs MiniMax H3 (via OpenVDN's 8-step VDN-H3 model) on consumer GPUs as a ComfyUI plugin, with a desktop launcher for Windows, macOS (preview) and Linux.
- **Platform:** desktop. The UI is a web front end inside ComfyUI (browser, `127.0.0.1`), plus a Qt/QML launcher window.
- **Who it's for:** hobbyists and creators with a gaming PC who want AI video without cloud credits or subscriptions. Chinese and English users (full zh-CN copy, QQ/WeChat groups).
- **Core loop people come for:** write a prompt, pick shape/length/quality, generate on your own GPU, preview, download. Free; there is no paid tier.
- **Slice for the replica:** the core loop (Create view + Creations library + launcher + engine). Everything else is should/could.

## 2. Sources

| # | Source | URL | What it gives |
|---|---|---|---|
| 1 | Repository README | https://github.com/FlashML-org/FreeVideo | Features, install paths, hardware floor, licence |
| 2 | Your fork | https://github.com/whiteak/FreeVideo-AI- | Same code, Apache 2.0 |
| 3 | Releases / changelog | https://github.com/FlashML-org/FreeVideo/releases | Version history (v0.1 to v0.6.13 in ~8 weeks) |
| 4 | Docs folder | https://github.com/FlashML-org/FreeVideo/tree/main/docs | Linux, Mac, LoRA, execution planner, community results |
| 5 | Bundled tutorial | `web/tutorial_book_content.js` (repo) | 44 how-to chapters with limits; best feature list |
| 6 | Feature tour | `web/feature_tour_content.js` (repo) | 19 headline features |
| 7 | Gallery | https://freevideo-community.pages.dev/#gallery | Sample outputs, quality levels side by side |
| 8 | Issues | https://github.com/FlashML-org/FreeVideo/issues | Bugs and requests (input for stage 8) |
| 9 | Model licence | https://huggingface.co/OpenVDN/vdn-minimax-h3-edge/blob/main/LICENSE | MiniMax H3 Community License: territorial and acceptable-use limits |
| 10 | Paper | https://arxiv.org/abs/2609.20744 | Video DeltaNet method |
| 11 | Discord | https://discord.gg/MsA277cJzZ | Community (not read) |

No pricing page exists; the product is free.

## 3. Screen inventory

Top bar on every web view: wordmark, view switcher (Create · Images · Nodes · Edit · Autopilot), status pill (Ready / busy, VRAM, GPU temp), search, Creations, phone/LAN, tags, tutorial book, feedback, settings. Reference screenshots in `replica/screens/` (from the repo's own tutorial images; reference only, never ship).

| ID | Screen | How reached | Purpose | Key components | States seen |
|---|---|---|---|---|---|
| S01 | Launcher: setup | Open FreeVideo.exe / .app / AppImage | Pick or install ComfyUI, add model folders, Install & launch | Folder pickers, download-source choice, progress meter, offline ZIP drop | first run, installing, offline import (docs) |
| S02 | Launcher: running | After launch | Status, open browser, release notes, close choice | Nav, release notes, QR code (LAN), exit/keep-running dialog | filled, close dialog (`launcher-close-2.png`) |
| S03 | Create (Studio) | Default view | Prompt to video | Prompt box, Media (frames/refs), @mention, shape/size, length stepper, quality (Light…Max), Enhance, LoRA, advanced, Generate button with time estimate, preview player, queue, recent works column | empty, filled (`quick-start-2.png`); generating/error *not seen* |
| S04 | Images | View switcher | Text/reference to image, layout boxes (MOSS) | Prompt, model picker, size, references (up to 16), history, compare | filled (`image-gen-2.png`); "Download model" state per docs |
| S05 | Image editor | From an image | Crop, retouch, text, AI edit of a selected area | Canvas, click-select (SAM2), shape list | per docs |
| S06 | Nodes | View switcher | Raw ComfyUI graph for LoRAs and custom workflows | ComfyUI canvas + FreeVideo top bar | inherited from ComfyUI |
| S07 | Edit | View switcher | Timeline editor: describe, extend, sound, text | Creations/Imported sidebar, player, per-part prompt + gen bar, toolbar (undo, cut, delete, record, marker, text, subtitles), Video/Audio tracks, beat snap, zoom, Export | filled (`editor-2.png`) |
| S08 | Export dialog | Edit → Export | Export content, quality, frame rate | Selects, progress, cancel | per docs |
| S09 | Autopilot | View switcher | Chat with an AI assistant that plans and shoots a multi-shot film | Chat, attachments, connection card, outline confirmation card, shot progress | connect state (`autopilot-2.png`) |
| S10 | Creations library | Top bar → Creations | Browse, tag, filter, compare, delete, export | All/Videos/Pictures/Favorites, tags, recently deleted, folder links, multi-select actions, side-by-side/slider compare | compare (`library-organize-2.png`) |
| S11 | Cast | Library → Cast tab | Characters, scenes, props for @mention | Cards, photo/sheet generate, face crop | per docs |
| S12 | Settings | Top bar gear | Model paths, AI assistant, privacy/system-status toggles, language | Settings kit groups | *not seen* |
| S13 | Run status / monitor | Status pill | VRAM, temperature, this computer, stop job | Meters, task list | thresholds per docs (red at 95% VRAM / 85 °C) |
| S14 | Tutorial book | Top bar book | 44 chapters with "Take me there" | Chapter list, media, steps, limits | per code |
| S15 | First-run guide, feature tour, what's new | First visit / after update | Onboarding | Cards with clips, try-it buttons | per code |
| S16 | Error panel / report issue | On failure | Explain failure, save anonymized diagnostic report | Error card, retry at lower quality, report | per docs |
| S17 | Share | From a result | Export share image/video stamped with gen time + GPU | Layout picker | per README |

## 4. User flows

| ID | Goal | Path | Happy-path clicks | Edge cases |
|---|---|---|---|---|
| F01 | First install | S01 → choose folder → Install & launch → S03 | 3 + waiting for multi-GB downloads | security software quarantine, offline packs, existing ComfyUI, partial downloads |
| F02 | **Make a video from text (core)** | S03: type prompt → (shape) → Generate → play → Download | **3** (Generate, play, Download) + typing | first run model load is slow; one job per GPU; out of memory → "retry at lower level" |
| F03 | Preview then finish | S03: Preview first → Run second pass | 2 | preview still needs model load |
| F04 | Video from first/last frames or references | S03: Media → add image(s) → prompt → Generate | 4–5 | Mac can't mix frames and references; center crop |
| F05 | Character reuse | S11: save character → S03: @name in prompt → Generate | ~5 | max 3 characters, 1 scene, 2 props |
| F06 | Long video / extend | S03 length > part size, or S07 select part → Extend | 2–3 | max 8 parts / 60 s; NVIDIA or Mac only; joins drift |
| F07 | Edit and export | S07: arrange, describe parts, add music/voice/text → Export → S08 | varies | used-in-edit videos can't be deleted; fonts missing glyphs |
| F08 | Autopilot film | S09: connect assistant → describe → confirm outline → wait | 3+ | no assistant connected; long reshoots ask first |
| F09 | Find and reuse | S10: pick creation → Reuse settings → S03 | 3 | videos without stored prompt |
| F10 | Troubleshoot | S16: read card → save report | 2 | all download mirrors fail |

## 5. Components

| Component | Variants / states | Used on |
|---|---|---|
| Top bar + view switcher | 5 views, narrow fit (collapses status to icon) | all web views |
| Status pill | Ready, busy, warning (red thresholds) | all |
| Prompt box | plain, with @mentions, timed lines, enhance on | S03, S04, S07 |
| Option chips (Media, @, shape, length stepper, quality, Enhance, LoRA, advanced) | default, active with count badge, disabled | S03, S07 |
| Generate button | idle with estimate, queued, running with progress, "Download model" | S03, S04, S07 |
| Video player (fv_player) | idle, playing, frame-step, loop, fullscreen | S03, S07, S10 |
| Thumbnail card | selected, duration badge, type badge | S07 sidebar, S10, S11 |
| Segmented control | 2–4 options | S07, S10, S01 |
| Dialog / popup | confirm, export, close-launcher | many |
| Error card | message, retry-lower, report | S16, inline |
| Timeline track + clip | video/audio, selected, generating, muted, locked | S07 |
| Toast / done notification | success, failure | all |
| Tooltip | plain, copyable | all |
| Launcher controls (QML F* set) | Button, Switch, Slider, Meter, ComboBox, Card, Nav, QR | S01, S02 |

## 6. Inferred data model

| Entity | Key fields | Relationships | Evidence | Confidence |
|---|---|---|---|---|
| Creation (video/image) | id, file path, kind, prompt, seed, size, duration, fps, quality, model, created_at, favorite | has many Tags; may come from a Task; may be used by an EditProject | library, reuse-settings, metadata-in-video (README v0.2.3) | high |
| Task (queue item) | id, status, progress, params, GPU, error | produces Creation(s) | queue, run status panel | high |
| Character / Scene / Prop (cast) | id, name, kind, images/sheet | mentioned by prompts via @ | tutorial cast chapters | high |
| LoRA | file, triggers, strength | many per Task (≤32) | LoRA chapter, `lora_*` modules | high |
| EditProject | id, name, tracks[] | Tracks have Clips; Clip → Creation, per-part prompt, text, audio | Edit view | high |
| Production (Autopilot film) | outline, shots, status | many Tasks; one EditProject | autopilot/drama chapters | medium |
| Tag | name (browser-local) | many Creations | "Tags are kept in this browser" | high |
| Settings | model paths, assistant, privacy, language | — | settings room | medium |

Storage is local files plus browser storage; there are no user accounts or server database.

## 7. Feature matrix

See `replica/features.csv`: 57 rows (14 must, 20 should, 23 could), 3 of them `skip`. `clone` is `no` everywhere to start. On the fork path, nearly every row is already present in your code and will flip to `yes` once checked on your PC.

## 8. What can't be cloned

| Item | Reason |
|---|---|
| MiniMax H3 / VDN-H3 weights | Third-party weights under the MiniMax H3 Community License (territorial + acceptable-use limits). Users download them; you can't relicense them. |
| Model hosting at `BowenXue/FreeVideo-models` on Hugging Face | Upstream author's repo. A fork still downloads from it unless you mirror (licence permitting). |
| Upstream update channel | `freevideo_update_keys.py` holds upstream's ed25519 release key and the updater points at `FlashML-org/FreeVideo` releases. A fork must swap both, or your users get upstream's updates. |
| Windows code-signing certificate | Belongs to upstream; you need your own or users hit SmartScreen warnings. |
| Community (Discord, QQ, WeChat, gallery site, community results) | Their network and users. |
| Name and logo "FreeVideo" | Apache 2.0 grants no trademark rights. Fine for personal use; a public release needs a new name. |

## 9. Size

| Path | Size | Why |
|---|---|---|
| Clean-room rebuild | **XL (rescope)** | ~188k lines of Python and ~72k lines of JS. The engine (low-VRAM streaming, int8/FP8 kernels, attention probing) is research-grade work. |
| Fork + rebrand + improvements | **M** | The code exists. Work is: rebrand sweep (5,408 "freevideo" mentions in 550 files, including the `freevideo_engine` package, `/freevideo/` routes and `fv-` CSS prefixes), own update key and release pipeline, own signing, then fixes from user feedback. |

**Three hardest parts**
1. The inference engine itself (hardware-adaptive, 8 GB VRAM floor). Rebuilding is out of reach; keep it.
2. Rebranding without breaking things: identifiers are everywhere, and ComfyUI workflows, saved videos' embedded metadata and node names depend on them.
3. Shipping your own releases: update signing, Windows signing, Mac notarization, and keeping up with an upstream that ships several releases a week.

**Out of scope:** the model weights, the community, upstream's signing and update infrastructure.
