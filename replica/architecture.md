# Architecture: your FreeVideo fork

Date: 2026-11-26. Path chosen: **improve the fork** (`whiteak/FreeVideo-AI-`), not a clean-room rebuild. So this stage keeps the existing stack and plans how to make the fork yours: cut the ties to upstream's infrastructure, rebrand what users see, ship your own builds, and keep pulling upstream's improvements.

## 1. Stack (kept as is)

| Layer | What it is | Why keep it |
|---|---|---|
| Inference engine | Python 3.12 package `freevideo_engine` (~188k lines), PyTorch + Triton, int8/FP8 weights, SageAttention probing | The product's whole value; rebuilding is out of reach |
| Host | ComfyUI custom node (`__init__.py`, `WEB_DIRECTORY = ./web`) | Gives the node graph, queue and model loaders for free |
| Web UI | Plain ES modules + CSS in `web/` (~72k lines), no build step | Edits show up on browser refresh; nothing to compile |
| Local API | ~150 aiohttp routes under `/freevideo/...` on ComfyUI's server (127.0.0.1) | Already complete |
| Launcher | PySide6 / QML window, packaged by PyInstaller (`scripts/build_windows.py`, `build_macos.py`, `build_linux.py`) | Installs ComfyUI, env and models in one click |
| Updates | `launcher_update.py` (exe/dmg updates) + `freevideo_source_bootstrap.py` (signed source overlays), both verified with an ed25519 key | Good design; needs your key and repo |
| CI | `.github/workflows/install.yml`: install checks on PRs; manual "release" run builds, signs (Azure Trusted Signing) and publishes | Reuse with your own secrets |
| Models | Downloaded on first run from Hugging Face / hf-mirror / ModelScope | Weights stay under their own licences |

There is no cloud backend, no accounts, no payments and no database. Stage 5 (backend) mostly doesn't apply.

## 2. Data (in place of schema.sql)

No `schema.sql`: the app keeps everything on the user's disk.

| Data | Where | Notes |
|---|---|---|
| Creations (videos, images) | ComfyUI `output/` folder | Prompt, seed and settings embedded in each file's metadata |
| Library state (favorites, undo, deleted) | Files beside the outputs, via `/freevideo/library/*` | Local only |
| Tags, UI preferences, last view | Browser `localStorage` | Lost if the browser profile is cleared |
| Edit projects | `/freevideo/editor/project*` files | Timeline tracks and clips |
| Characters, scenes, props | `/freevideo/agent/cast*` | Used by @mentions |
| Settings, launcher state | Launcher settings + `/freevideo/output-settings`, `/freevideo/language` | |
| Models | ComfyUI `models/` (+ any folders the user adds) | Many GB; reused across installs |

If you later add anything online (accounts, a gallery, paid cloud renders), that's when a real schema gets written.

## 3. Local API map (by flow)

All routes are local (127.0.0.1, or the LAN when `server --listen 0.0.0.0` is used). Callers are the web UI and, for `/freevideo/agent/*`, external AI agents.

| Group | Example routes | Flow |
|---|---|---|
| Generate + queue | ComfyUI's own `POST /prompt` (the Create view submits a workflow), plus `/freevideo/jobs`, `/queue/first`, `/progress`, `/cancel`, `/sampling-estimate` | F02, F03 |
| Long video chains | `/chain/plan`, `/chain/{id}/resume`, `/chain/{id}/blocks/{block}/redo` | F06 |
| Library | `/library`, `/library/delete-batch`, `/library/download-zip`, `/library/favorite`, `/library/undo` | F09 |
| Editor | `/editor/project`, `/editor/export`, `/editor/music`, `/editor/speech`, `/editor/subtitles`, `/editor/beats` | F07 |
| Images | `/image-gen/{action}` | S04, S05 |
| Agents / Autopilot | `/agent/chat/*`, `/agent/connect`, `/agent/v1/jobs/`, `/agent/cast/*`, MCP in `agent/mcp.py` | F05, F08 |
| Setup + models | `/setup/{action}`, `/launcher/models`, `/fast-pack`, `/swap-lora`, `/request-assets` | F01 |
| Health + support | `/monitor`, `/executor/hardware`, `/report/{id}`, `/feedback-info` | F10 |
| Updates | `/updates`, `/updates/apply` | (upstream-coupled, see §4) |

Webhooks: none. Background work: the generation queue (one job per GPU), model downloads, update checks, cache cleanup (`/private/*`).

## 4. Upstream ties to cut (the parts that bite)

A URL scan found no analytics or tracking hosts; outbound traffic is model downloads, update checks and links. These are the places that still point at upstream:

| # | Where | Today | Change to | Needed for |
|---|---|---|---|---|
| U1 | `freevideo_engine/launcher_update.py` `REPOSITORY`, `MIRROR_BASE` | `FlashML-org/FreeVideo` releases; ModelScope `BowenXue/FreeVideo-releases` | Your repo's releases (and no mirror, or your own) | Any fork you install with the launcher |
| U2 | `freevideo_update_keys.py` | Upstream's ed25519 public key | Your own public key (private key only in a CI secret) | Your own releases |
| U3 | `freevideo_source_bootstrap.py` line ~214 | Only trusts overlays from `FlashML-org/FreeVideo` | Your repo | Your own releases |
| U4 | `.github/workflows/install.yml`, `scripts/publish_release.py`, `scripts/build_*.py` | Signing account `flashmlsigning`, upstream repo names | Your signing (or unsigned builds to start), your repo | Your own releases |
| U5 | `web/feedback.js`, `web/health.js`, `web/issue_button.js`, `web/report_issue.js`, `launcher_issue.py`, `.github/ISSUE_TEMPLATE/config.yml` | Bug reports go to upstream's issues | Your repo's issues | Public release (don't send your users' bugs to them) |
| U6 | `model_shares.json`, `swap_lora.json`, `fast_pack.json`, `image_gen/image_models.json` | Some model packs come from `BowenXue/...` on HF/ModelScope and a Quark share | Keep for now (weights licence permitting); mirror later if they vanish | Resilience |
| U7 | README, launcher `Main.qml`, community links | Discord, QQ, WeChat, gallery | Your own links or none | Public release |

**Important for personal use, today:** if you install your fork with upstream's launcher, U1–U3 mean an update can replace your changes with upstream's code. Until U1–U3 are done, run from source (`setup.cmd` then `freevideo.cmd`, or clone your fork into an existing ComfyUI's `custom_nodes`) and don't accept update offers. There is no setting to switch update checks off, so M1 is where that gets fixed.

Other things to keep in mind:
- **Upstream moves fast** (v0.1 to v0.6.13 in under 8 weeks, often several releases a week). Every file you touch is a future merge conflict.
- **Identifiers are load-bearing.** Saved workflows, metadata inside existing videos, ComfyUI node class names and `/freevideo/` routes all depend on the internal names.
- **Windows signing:** unsigned .exe files trigger SmartScreen and antivirus false positives (upstream has a whole scan step for this).
- **Model licence:** the MiniMax H3 Community License restricts territory and use; a public release must pass its terms through to users.

## 5. Rebrand strategy: surface only

Rebrand what people see, and leave internal identifiers alone.

- **Change:** product name in UI copy, window titles, launcher, exe/app names, icons, logo SVGs (`web/assets/freevideo*.svg`, `wordmark.js`), README, release notes, links (U5, U7).
- **Keep:** the `freevideo_engine` package, `/freevideo/` routes, `fv-` CSS classes, ComfyUI node names, metadata keys.
- **Why:** Apache 2.0 only needs the licence and NOTICE kept, and trademark only cares about what users see. Renaming the 5,408 internal mentions would break saved workflows and make every upstream merge painful.
- **How:** put the display name, links and logo paths in one place (a small `brand` module for Python and one for JS), and point the visible strings at it. That makes the rebrand a short diff that merges cleanly.
- Stage 9's sweep then covers user-visible text, filenames of shipped assets, titles and icons, not internal identifiers.

Only needed if you go public. For personal use, skip §5 and do U1–U3.

## 6. Upstream sync

- Add upstream as a remote: `git remote add upstream https://github.com/FlashML-org/FreeVideo.git`.
- Keep your changes on `main` in small, separate commits (U1–U7, brand module, each fix).
- Merge `upstream/main` weekly, or before each of your releases. Run `test.cmd` and a short generation before tagging.
- When upstream fixes something you also changed, take theirs.

## 7. Build order

| Milestone | What | Size | Files |
|---|---|---|---|
| M0 | Run your fork unchanged on your PC from source; one 5 s Light video end to end | S | none |
| M1 | Cut update ties U1–U3 so updates can't overwrite your code (point at your repo, or disable) | S | `launcher_update.py`, `freevideo_update_keys.py`, `freevideo_source_bootstrap.py` |
| M2 | Your own release pipeline: CI on your repo, unsigned builds first, your update key as a secret (U4) | M | `.github/workflows/install.yml`, `scripts/*` |
| M3 | Improvements from user feedback (stage 8 picks them) | M | varies |
| M4 | Public only: brand module + surface rebrand (§5), U5–U7, own signing | M | `web/`, launcher QML, assets, README |
| M5 | Public only: launch + deploy (stages 10–11) | S–M | docs, releases |

Stage 3 (design) and stage 4 (build) shrink on this path: the design system already exists (`web/tokens.css`, `web/palette.css`, `freevideo_engine/branding.py`); stage 3 becomes a contrast check and a new accent colour for M4.
