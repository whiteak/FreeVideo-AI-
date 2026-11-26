# Fix plan for your fork

Date: 2026-11-26. Picked from `replica/feedback.md`, by how often users hit it × how cheap it is × whether it matters on **your** PC (Windows, RTX 5080 16 GB). That leaves out the biggest themes, AMD support (H2), older GPUs (M1) and installer failures (H1), because they don't affect you and are large jobs.

Sizes: S = an evening, M = a few days. "Stage" is the replica stage that does the work.

| # | Fix | Evidence | Size | Stage | Where in the code |
|---|---|---|---|---|---|
| X1 | **Update when you choose.** A one-click script that pulls upstream's latest release into your fork, shows what changed, and keeps your own changes on top | Upstream updates broke things people relied on (H5; [#678](https://github.com/FlashML-org/FreeVideo/issues/678), [#966](https://github.com/FlashML-org/FreeVideo/issues/966)); your fork no longer auto-updates, so it needs a way to catch up | S | 4 build | new `scripts/update-from-upstream.ps1` |
| X2 | **No-subtitles / no-music / no-narration switches** in Create that add a firm instruction to the prompt | ~21 issues (H6), e.g. [#1091](https://github.com/FlashML-org/FreeVideo/issues/1091), [#521](https://github.com/FlashML-org/FreeVideo/issues/521). The engine has no negative prompt, so this is an experiment: we test whether H3 follows it. | S | 4 build | Create panel prompt assembly (`web/studio*.js`) |
| X3 | **Settings that stick:** LoRAs, references and a locked seed stay set after a run and after a restart | ~9 issues (M3), e.g. [#1025](https://github.com/FlashML-org/FreeVideo/issues/1025), [#766](https://github.com/FlashML-org/FreeVideo/issues/766) | S–M | 4 build | `web/workspace_restore.js`, `web/workflow_drafts.js` |
| X4 | **Memory between runs:** a setting to either free VRAM after each video, or keep models loaded when RAM allows so back-to-back videos skip the reload | ~59 issues (H4), e.g. [#838](https://github.com/FlashML-org/FreeVideo/issues/838), [#442](https://github.com/FlashML-org/FreeVideo/issues/442), [#750](https://github.com/FlashML-org/FreeVideo/issues/750). Your test run peaked at 14.7 of 16 GB, so this matters on your card. | M | 4 build + 6 test | `freevideo_engine/comfy_residency.py`, settings room |
| X5 | **More aspect ratios:** 4:3, 3:4, 3:2, 2:3 and 21:9 in Create | 1 issue (thin), [#844](https://github.com/FlashML-org/FreeVideo/issues/844); the image editor already has 4:3 and 3:4 | S | 4 build | `web/frame_sizes.js` (3 lines today). 21:9 needs a quality test, since the model may not be trained on it. |
| X6 | **Creation details:** show the model, size and seed on every image/video card, a "copy file path" button, and save images to the output folder instead of input | ~60 issues on organising work (M2), e.g. [#819](https://github.com/FlashML-org/FreeVideo/issues/819), [#939](https://github.com/FlashML-org/FreeVideo/issues/939), [#348](https://github.com/FlashML-org/FreeVideo/issues/348) | S | 4 build | library cards (`web/work_column.js`, `web/library*.js`), image save path |
| X7 | **Use the whole screen:** a wide layout for the Create panel on big monitors, and Creations always visible | ~16 issues, 13 in the last 2 weeks (H5), e.g. [#887](https://github.com/FlashML-org/FreeVideo/issues/887), [#875](https://github.com/FlashML-org/FreeVideo/issues/875) | M | 3 design + 4 build | `web/studio_columns.js`, `web/studio.css` |

**Recommended first batch:** X1, X2 and X3. They're small, they fix things you'll hit soon (catching up with upstream, unwanted subtitles and music, re-entering settings), and none of them touches the engine.

These are added to `replica/features.csv` with `original=no`.

## Not planned, and why

- AMD / Intel / older NVIDIA support: not your hardware, and the engine's kernels target Ampere and newer.
- Installer and antivirus fixes: you run from source, so you skip the launcher entirely.
- Lip sync, new languages: thin evidence, large work.
