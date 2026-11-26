"""LoRA Studio check against a running FreeVideo (fork addition). Uses the real picture and prompt models.

    envs\\unified\\python.exe scripts\\lora_studio_smoke.py [--url http://127.0.0.1:8188] [--keep] [--slots 2]

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
    parser.add_argument('--slots', type=int, default=2)
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()
    client = Client(args.url)

    t = step('LoRA Studio info')
    info = client.call('GET', '/freevideo/lora-studio/api/info')
    done(t, 'trainer ready=%s h3=%s' % (info['trainer']['ready'], info['trainer']['h3']))
    t = step('Picture models')
    models = {row['id']: row.get('ready') for row in client.call('GET', '/freevideo/image-gen').get('models', [])}
    model = 'minimax-h3-image' if models.get('minimax-h3-image') else 'qwen-image-2.1-turbo' if models.get('qwen-image-2.1-turbo') else None
    done(t, json.dumps(models))
    if not model:
        sys.exit('No picture model that edits from a reference is ready.')

    t = step('Create a character set')
    dataset = client.studio('create', dict(name='Smoke test', kind='character', class_word='woman',
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

        for slot in SLOTS[:args.slots]:
            t = step('Slot %s' % slot)
            plan = client.studio('slot-prompt', dict(dataset=ds, slot=slot))
            staged = client.studio('stage', dict(dataset=ds, item=reference['id']))['file']
            file = make_picture(client, model, plan, [staged])
            result = client.studio('adopt', dict(dataset=ds, file=file, slot=slot, prompt=plan['prompt'], model=model))
            done(t, result['item']['input'])

        items = [row for row in client.studio('get', dict(dataset=ds))['items'] if not row['excluded']]
        questions = client.studio('questions', dict(dataset=ds))
        t = step('Captions for %d pictures' % len(items))
        answers = ask(client, [dict(text=questions['caption'], images=[dict(file=row['input'])]) for row in items[:8]],
                      system=questions['caption_system'], max_tokens=160)
        for row, answer in zip(items, answers):
            client.studio('caption', dict(dataset=ds, item=row['id'], text=answer.get('text', '')))
            print('   %s: %s' % (row.get('slot') or 'reference', answer.get('text', answer)))
        done(t)

        others = [row for row in items if row['id'] != reference['id']]
        if others:
            t = step('Likeness check for %d pictures' % len(others))
            answers = ask(client, [dict(text=questions['drift'], images=[dict(file=reference['input'], label='reference'),
                                                                        dict(file=row['input'], label='candidate')])
                                   for row in others[:8]], system=questions['drift_system'], format='json', max_tokens=200)
            for row, answer in zip(others, answers):
                saved = client.studio('drift', dict(dataset=ds, item=row['id'], answer=answer))
                drift = next(r for r in saved['items'] if r['id'] == row['id'])['drift']
                print('   %s: %s' % (row['slot'], drift))
            done(t)

        t = step('Export ai-toolkit layout')
        exported = client.studio('export', dict(dataset=ds, layout='ai-toolkit'))
        done(t, exported['path'])
        t = step('Training plan')
        plan = client.studio('train-plan', dict(dataset=ds))
        done(t, 'warnings: %s' % plan['warnings'])
        print('\nALL STEPS PASSED')
    finally:
        if not args.keep:
            client.studio('delete', dict(dataset=ds))
            print('(test set deleted; run with --keep to look at it in LoRA Studio)')


if __name__ == '__main__':
    main()
