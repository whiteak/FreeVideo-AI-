"""Hardlink picture, music, MOSS and prompt model files from another FreeVideo install into this one.

`setup --reuse-models` only links the video models. The picture models (H3 Image, Qwen Image, MOSS, music) are
downloaded later from the app, as is the prompt model (Qwen3-VL) that writes prompts and captions, so a second install on the same PC would download tens of GB it already has.

This links each catalog file that is missing here and has the exact expected size in the other install. Nothing is
copied or overwritten. The app still checks every file by its SHA-256 before using it and downloads only what fails.

    python scripts/reuse_image_models.py <other install's model folder> [--dry-run]

The other folder can be the install folder, its models folder or its models\\vdn folder.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freevideo_engine.image_gen import image_pack  # noqa: E402


def source_root(folder):
    """The other install's model folder (the one holding image_gen)."""
    folder = Path(folder).expanduser().resolve()
    for candidate in (folder, folder / 'vdn', folder / 'models' / 'vdn'):
        if (candidate / 'image_gen').is_dir():
            return candidate
    raise SystemExit(f'No image_gen folder found under {folder}. Point this at the other install\'s models\\vdn folder.')


def source_path(row, root):
    relative = Path(*Path(row['file']).parts)
    if row.get('shared'):                       # the text encoder lives beside the video encoders
        return root.parent / 'encoder' / relative
    return root / 'image_gen' / relative


def gb(n):
    return f'{n / 1e9:.1f} GB'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('other', help="the other install's model folder")
    parser.add_argument('--dry-run', action='store_true', help='only show what would be linked')
    args = parser.parse_args(argv)

    other = source_root(args.other)
    here = image_pack.base().parent
    if os.path.normcase(str(other)) == os.path.normcase(str(here.resolve())):
        raise SystemExit('That is this install\'s own model folder.')
    print(f'From: {other}\nTo:   {here}\n')

    linked = present = missing = different = failed = 0
    saved = 0
    seen = set()
    for row in image_pack.rows(group=None):
        mine = image_pack.target(row)
        if str(mine) in seen:
            continue
        seen.add(str(mine))
        theirs = source_path(row, other)
        if mine.exists():
            present += 1
            continue
        if not theirs.is_file():
            missing += 1
            continue
        if theirs.stat().st_size != row['bytes']:
            different += 1
            print(f'  skip (different version): {row["file"]}')
            continue
        if args.dry_run:
            print(f'  would link {gb(row["bytes"]):>8}  {row["file"]}')
        else:
            try:
                mine.parent.mkdir(parents=True, exist_ok=True)
                os.link(theirs, mine)
            except OSError as error:
                failed += 1
                print(f'  could not link {row["file"]}: {error.strerror or error}')
                continue
            print(f'  linked {gb(row["bytes"]):>8}  {row["file"]}')
        linked += 1
        saved += row['bytes']

    # The prompt model (Qwen3-VL) writes prompts and captions; it lives in <install>/models/prompt-vlm/<revision>.
    try:
        from freevideo_engine.paths import data_root
        from freevideo_engine.prompt_vlm import assets as prompt_assets
        spec = prompt_assets.catalog()
        theirs_dir = other.parent / 'prompt-vlm' / spec['revision']
        mine_dir = prompt_assets.directory(data_root())
    except (ImportError, OSError, ValueError, KeyError):
        spec = None
    if spec and theirs_dir.is_dir() and os.path.normcase(str(theirs_dir.resolve())) != os.path.normcase(str(mine_dir.resolve())):
        for row in spec['files']:
            mine, theirs = mine_dir / row['file'], theirs_dir / row['file']
            if mine.exists():
                present += 1
                continue
            if not theirs.is_file():
                missing += 1
                continue
            if theirs.stat().st_size != row['bytes']:
                different += 1
                print(f'  skip (different version): prompt-vlm/{row["file"]}')
                continue
            if args.dry_run:
                print(f'  would link {gb(row["bytes"]):>8}  prompt-vlm/{row["file"]}')
            else:
                try:
                    mine.parent.mkdir(parents=True, exist_ok=True)
                    os.link(theirs, mine)
                except OSError as error:
                    failed += 1
                    print(f'  could not link prompt-vlm/{row["file"]}: {error.strerror or error}')
                    continue
                print(f'  linked {gb(row["bytes"]):>8}  prompt-vlm/{row["file"]}')
            linked += 1
            saved += row['bytes']
        # The app trusts the folder only with its receipt (verified.json: revision and each file's modified time).
        # Hardlinks share the files' times, so the other install's receipt holds here too, once every file is the
        # very same file; anything else is left for the app to check and download.
        receipt = theirs_dir / 'verified.json'
        try:
            same = all(os.path.samefile(mine_dir / row['file'], theirs_dir / row['file']) for row in spec['files'])
        except OSError:
            same = False
        if not args.dry_run and same and receipt.is_file() and not (mine_dir / 'verified.json').exists():
            try:
                import json
                if json.loads(receipt.read_text(encoding='utf-8')).get('revision') == spec['revision']:
                    shutil.copy2(receipt, mine_dir / 'verified.json')
                    print('  copied the prompt model\'s receipt (verified.json): it is ready without checking again')
            except (OSError, ValueError) as error:
                print(f'  could not copy the prompt model\'s receipt: {error}')

    verb = 'Would link' if args.dry_run else 'Linked'
    print(f'\n{verb} {linked} files ({gb(saved)}). Already here: {present}. Not in the other install: {missing}.'
          f' Different version: {different}.' + (f' Failed: {failed} (both folders must be on the same drive).' if failed else ''))
    if linked and not args.dry_run:
        print('Open FreeVideo and start the picture model download as usual: it checks these files and fetches only the rest.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
