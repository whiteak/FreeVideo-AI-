"""LoRA Studio check against a running FreeVideo (fork addition). Uses the real picture and prompt models.

    envs\\unified\\python.exe scripts\\lora_studio_smoke.py [--url http://127.0.0.1:8188] [--keep] [--slots 2]
    envs\\unified\\python.exe scripts\\lora_studio_smoke.py --slots all --name Ava --train   (a whole set, then training)

Creates a character set from a description, makes the reference with H3 Image, fills a few coverage slots as edits of
it, captions every picture and runs the likeness check with the prompt model, then exports the ai-toolkit layout.
Prints each step with its time. Run it while nothing else is generating. Deletes the set afterwards unless --keep.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SLOTS = ['face_front', 'three_quarter_left', 'smile', 'walking', 'night_neon', 'back']


class Client:
    def __init__(self, url):
        self.url = url.rstrip('/')
        self.tokens = {}

    def call(self, method, path, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode('utf-8')
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read() or b'{}')
        except urllib.error.HTTPError as error:
            text = error.read().decode('utf-8', 'replace')
            raise RuntimeError('%s %s -> %d %s' % (method, path, error.code, text[:500])) from None

    def token(self, name, path):
        if name not in self.tokens:
            self.tokens[name] = self.call('GET', path)['token']
        return self.tokens[name]

    def studio(self, action, body=None):
        return self.call('POST', '/freevideo/lora-studio/api/' + action, body or {},
                         {'X-FreeVideo-Lora': self.token('studio', '/freevideo/lora-studio/api/info')})

    def image(self, action, body):
        return self.call('POST', '/freevideo/image-gen/' + action, body,
                         {'X-FreeVideo-Image': self.token('image', '/freevideo/image-gen')})

    def vlm(self, action, body):
        return self.call('POST', '/freevideo/prompt-vlm/' + action, body,
                         {'X-FreeVideo-Prompt': self.token('vlm', '/freevideo/prompt-vlm')})


def step(name):
    print('\n== %s' % name, flush=True)
    return time.monotonic()


def done(started, detail=''):
    print('   ok in %.1f s %s' % (time.monotonic() - started, detail), flush=True)


def make_picture(client, model, plan, images):
    started = client.image('generate', dict(model=model, prompt=plan['prompt'], width=plan['width'], height=plan['height'],
                                            images=images, seed=int(time.time()) % 100000))
    last = ''
    while True:
        time.sleep(1)
        status = client.image('status', dict(job=started['job']))
        phase = status.get('phase')
        if phase == 'complete':
            return status['file']
        if phase in ('failed', 'cancelled'):
            raise RuntimeError('picture failed: %s' % json.dumps(status)[:800])
        line = '%s %s %.0f%%' % (phase, status.get('stage', ''), 100 * (status.get('fraction') or 0))
        if line != last:
            print('   ' + line, flush=True)
            last = line


def plan_impostor(client, ds):
    """The reference's size and wording, with someone else in the same clothes."""
    plan = client.studio('slot-prompt', dict(dataset=ds, slot='anchor'))
    prompt = plan['prompt'].replace('a woman in her thirties with short curly red hair and freckles',
                                    'a woman in her fifties with long straight black hair and no freckles')
    return dict(prompt=prompt, width=plan['width'], height=plan['height'])


def ask(client, questions, **options):
    started = client.vlm('ask', dict(questions=questions, **options))
    while True:
        time.sleep(1)
        status = client.vlm('status', dict(job=started['job']))
        if status.get('phase') == 'complete':
            return status.get('answers') or []
        if status.get('phase') in ('failed', 'cancelled'):
            raise RuntimeError('prompt model failed: %s' % json.dumps(status)[:800])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8188')
    parser.add_argument('--slots', default='2', help='how many coverage slots to fill, or "all"')
    parser.add_argument('--name', default='Smoke test')
    parser.add_argument('--train', action='store_true',
                        help='after the checks, leave out drifted pictures and start training (keeps the set)')
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--impostor', action='store_true',
                        help='also make a different person in the same clothes and check the likeness check flags them')
    parser.add_argument('--only', default='', help='comma-separated slot ids to make instead of the first --slots')
    args = parser.parse_args()
    if args.train:
        args.keep = True
    # A Windows console is cp1252 by default; the side and back prompts are Chinese.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    client = Client(args.url)

    t = step('LoRA Studio info')
    info = client.call('GET', '/freevideo/lora-studio/api/info')
    done(t, 'trainer ready=%s h3=%s' % (info['trainer']['ready'], info['trainer']['h3']))
    if args.train and not info['trainer']['ready']:
        sys.exit('--train needs the trainer: run scripts\\install-lora-trainer.ps1 first. %s' % json.dumps(info['trainer']))
    t = step('Picture models')
    models = {row['id']: row.get('ready') for row in client.call('GET', '/freevideo/image-gen').get('models', [])}
    model = 'minimax-h3-image' if models.get('minimax-h3-image') else 'qwen-image-2.1-turbo' if models.get('qwen-image-2.1-turbo') else None
    done(t, json.dumps(models))
    if not model:
        sys.exit('No picture model that edits from a reference is ready.')

    t = step('Create a character set')
    dataset = client.studio('create', dict(name=args.name, kind='character', class_word='woman',
                                           description='a woman in her thirties with short curly red hair and freckles, '
                                                       'green parka, blue jeans, white trainers'))
    ds = dataset['id']
    done(t, ds)
    try:
        t = step('Reference from the description (%s)' % model)
        plan = client.studio('slot-prompt', dict(dataset=ds, slot='anchor'))
        file = make_picture(client, model, plan, [])
        result = client.studio('adopt', dict(dataset=ds, file=file, prompt=plan['prompt'], model=model, source='generated'))
        reference = result['item']
        client.studio('update', dict(dataset=ds, reference=reference['id']))
        done(t, reference['input'])

        if args.slots == 'all':
            every = [row['id'] for row in info['plans']['character']]
        else:
            every = SLOTS[:int(args.slots)]
        for slot in ([s.strip() for s in args.only.split(',') if s.strip()] or every):
            t = step('Slot %s' % slot)
            plan = client.studio('slot-prompt', dict(dataset=ds, slot=slot))
            print('   prompt: ' + plan['prompt'][:300], flush=True)
            staged = client.studio('stage', dict(dataset=ds, item=reference['id']))['file']
            try:
                file = make_picture(client, model, plan, [staged])
            except RuntimeError as error:
                if not args.train:
                    raise
                print('   skipped: %s' % error, flush=True)  # one failed picture shouldn't stop a long unattended run
                continue
            result = client.studio('adopt', dict(dataset=ds, file=file, slot=slot, prompt=plan['prompt'], model=model))
            done(t, result['item']['input'])

        items = [row for row in client.studio('get', dict(dataset=ds))['items'] if not row['excluded']]
        questions = client.studio('questions', dict(dataset=ds))
        t = step('Captions for %d pictures' % len(items))
        answers = []
        for start in range(0, len(items), 8):  # the prompt model takes 8 questions at a time
            answers += ask(client, [dict(text=questions['caption'], images=[dict(file=row['input'])])
                                    for row in items[start:start + 8]], system=questions['caption_system'], max_tokens=160)
        for row, answer in zip(items, answers):
            client.studio('caption', dict(dataset=ds, item=row['id'], text=answer.get('text', '')))
            print('   %s: %s' % (row.get('slot') or 'reference', answer.get('text', answer)))
        done(t)

        others = [row for row in items if row['id'] != reference['id']]
        if others:
            t = step('Likeness check for %d pictures' % len(others))
            answers = []
            for start in range(0, len(others), 8):
                answers += ask(client, [dict(text=questions['drift'], images=[dict(file=reference['input'], label='reference'),
                                                                             dict(file=row['input'], label='candidate')])
                                        for row in others[start:start + 8]], system=questions['drift_system'],
                               format='json', max_tokens=200)
            for row, answer in zip(others, answers):
                saved = client.studio('drift', dict(dataset=ds, item=row['id'], answer=answer))
                drift = next(r for r in saved['items'] if r['id'] == row['id'])['drift']
                print('   %s: %s' % (row['slot'], drift))
                if args.train and drift.get('score') is not None and drift['score'] < info['drift_limit']:
                    client.studio('item-update', dict(dataset=ds, item=row['id'], excluded=True))
                    print('      left out of training (likeness below %s)' % info['drift_limit'])
            done(t)

        if args.impostor:
            t = step('Likeness check on a different person (should score below 6)')
            other = make_picture(client, model, dict(plan_impostor(client, ds)), [])
            answers = ask(client, [dict(text=questions['drift'], images=[dict(file=reference['input'], label='reference'),
                                                                        dict(file=other, label='candidate')])],
                          system=questions['drift_system'], format='json', max_tokens=200)
            print('   %s: %s' % (other, answers[0].get('json') or answers[0].get('text')))
            done(t)

        t = step('Export ai-toolkit layout')
        exported = client.studio('export', dict(dataset=ds, layout='ai-toolkit'))
        done(t, exported['path'])
        t = step('Training plan')
        plan = client.studio('train-plan', dict(dataset=ds))
        done(t, '%d pictures, about %s h, warnings: %s' % (plan['pictures'], plan['estimate_hours'], plan['warnings']))
        if args.train:
            t = step('Start training')
            started = client.studio('train-start', dict(dataset=ds))
            done(t, json.dumps(started)[:400])
            print('Training runs inside FreeVideo; watch it in LoRA Studio > Train. Set %s.' % ds)
        print('\nALL STEPS PASSED')
    finally:
        if not args.keep:
            client.studio('delete', dict(dataset=ds))
            print('(test set deleted; run with --keep to look at it in LoRA Studio)')


if __name__ == '__main__':
    main()
