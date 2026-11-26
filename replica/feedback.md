# What FreeVideo users struggle with

Date: 2026-11-26. Feeds stage 8 (entrepreneur), adapted for a **personal** fork: the aim is improvements worth having on your own PC, not a market position.

## Sources and how they were read

| Source | What | Rows |
|---|---|---|
| GitHub issues, FlashML-org/FreeVideo | All 941 issues (2026-10-08 to 2026-11-26), via GitHub's public search API | 673 used. The 268 `[Report]` issues are opt-in benchmark submissions (GPU, size, time), not complaints, so they're left out of the themes. |
| Hacker News thread on native H3 inference for Apple Silicon | https://hn.svelte.dev/item/49252179 | About MiniMax H3 locally in general, not FreeVideo. Used only for context on speed. |
| Hugging Face discussion "Strix Halo 8060s" | https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/33 | H3 on AMD hardware. Context for the AMD theme. |

No app store, G2 or Trustpilot listing exists for FreeVideo, and no Reddit thread about it turned up in search. **The evidence is close to single-source (GitHub issues).** Many issues are pre-filled error reports sent from the app's "Report a problem" button, so they show what *breaks*, more than what people *wish for*. Nearly half are in Chinese.

Counts come from keyword matching over each issue's title and text. An issue can count toward two themes, so treat the numbers as rough sizes, not exact tallies. "Last 2 weeks" means since 2026-11-12. Full rows are in `replica/reviews.csv` (one per issue, with link and date; GitHub issues have no star rating).

## What they hate

| # | Theme | Issues | Last 2 wks | Example (verbatim, linked) |
|---|---|---|---|---|
| H1 | **Installer failures** (step couldn't complete, downloads interrupted, folder not writable, Smart App Control, Git setup) | ~258 | ~148 | "Couldn’t complete this step" is the 2nd most common launcher error (45 issues). [#195](https://github.com/FlashML-org/FreeVideo/issues/195): "Windows Smart App Control Blocks PyTorch DLLs" |
| H2 | **"No NVIDIA driver was found"** and AMD users locked out | ~102 | ~76 | 84 issues titled exactly "No NVIDIA driver was found". [#1008](https://github.com/FlashML-org/FreeVideo/issues/1008): "Cant install on AMD video cards!!!" |
| H3 | **Generation didn't finish** (crashes mid-job, missing Fast-mode files, queue errors) | ~74 | ~23 | [#1156](https://github.com/FlashML-org/FreeVideo/issues/1156): "FAST_MODE_FILES_MISSING: install the fastest mode files" |
| H4 | **Memory: VRAM not freed between runs, or models reloaded every run** | ~59 | ~28 | [#838](https://github.com/FlashML-org/FreeVideo/issues/838): "上个视频生成后没有清除显存占用导致生成新的视频会出现爆显存的情况" (VRAM isn't released after a video, so the next one runs out). [#442](https://github.com/FlashML-org/FreeVideo/issues/442): the opposite complaint, models unloaded after every video and reloaded from disk. |
| H5 | **The 0.6 redesign of the Create panel** (laggy, crowded, hides Creations, wastes widescreens) | ~16 | ~13 | [#887](https://github.com/FlashML-org/FreeVideo/issues/887): "On an ultrawide monitor (5120x1440) this wastes more than half of the screen". [#966](https://github.com/FlashML-org/FreeVideo/issues/966): "Make Roll Back Easier" |
| H6 | **Unwanted subtitles, background music or narration** that prompts can't switch off | ~21 | ~8 | [#1091](https://github.com/FlashML-org/FreeVideo/issues/1091): "我不想要自动生成的字幕，如何设置才能不自动生成字幕?" (How do I stop auto-generated subtitles?). [#521](https://github.com/FlashML-org/FreeVideo/issues/521): the prompt says no background music, it's added anyway. |
| H7 | **Antivirus flags the launcher** | ~7 | ~4 | [#839](https://github.com/FlashML-org/FreeVideo/issues/839): "antivirus is showing it as Trojen" |

## What's missing

| # | Theme | Issues | Example |
|---|---|---|---|
| M1 | **Older GPU support** (GTX 10/16, RTX 20) | ~54 | [#317](https://github.com/FlashML-org/FreeVideo/issues/317): "Add old GPUs support" |
| M2 | **Organising work: projects, batch download, model info on cards, copy file path** | ~60 | [#1131](https://github.com/FlashML-org/FreeVideo/issues/1131): "projects/workspaces to group assets, history and set…". [#819](https://github.com/FlashML-org/FreeVideo/issues/819): "I can't see which model generated which image." |
| M3 | **Settings that stick between runs** (LoRAs, references, seed) | ~9 | [#1025](https://github.com/FlashML-org/FreeVideo/issues/1025): "После каждого запуска сбрасываются настроенные на сцену Lora и референсы." (LoRAs and references reset after every run). [#766](https://github.com/FlashML-org/FreeVideo/issues/766): seed stopped staying fixed after an update. |
| M4 | **More aspect ratios** | thin (1) | [#844](https://github.com/FlashML-org/FreeVideo/issues/844): asks for 2:3, 3:2, 3:4, 4:3 and 21:9 alongside 16:9, 9:16 and 1:1 |
| M5 | **Lip sync / exact audio** | ~4 (thin) | [#1004](https://github.com/FlashML-org/FreeVideo/issues/1004): "Do you have any plans to add a "lip sync" feature?" |
| M6 | **Russian / Portuguese interface** | ~4 (thin) | [#143](https://github.com/FlashML-org/FreeVideo/issues/143), [#1056](https://github.com/FlashML-org/FreeVideo/issues/1056) |

## What's unsolved

- **People without a supported NVIDIA card.** H2 and M1 together are the largest group that can't use FreeVideo at all. The HF thread shows AMD owners getting H3 working in plain ComfyUI with ROCm and tuning. Not relevant to your RTX 5080.
- **Prompt control over the soundtrack and on-screen text.** H3 generates audio and sometimes text itself, and there's no switch for either (H6, M5).
- **Upstream's pace.** Updates often break things people relied on (H5; reference-video length cut in v0.4.30, [#678](https://github.com/FlashML-org/FreeVideo/issues/678)). A personal fork that updates only when *you* choose is already a fix for this.

## Thin or single-source themes

M4, M5 and M6 rest on 1 to 4 issues. Everything rests on one source (GitHub issues), with Hacker News and Hugging Face only adding context.
