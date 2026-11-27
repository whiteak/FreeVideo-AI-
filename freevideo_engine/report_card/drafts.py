"""Draft small, finish the winner (plan step 1.2): several seeds made as half-size first-pass previews, each checked
by the Report Card, and only the best one finished at full size. When no draft passes, one more round of drafts is
made before the best of everything is finished.

It drives the agent's own jobs (agent/jobs.py: takes, preview, refine_from), so the queue, the GPU turn-taking and the
library work as for any other request.
"""
import threading
import time
import uuid

POLL = 2.
WAIT = 6 * 3600
ROUNDS = 2


class Runs:
    """Draft runs of this server session, by id; each runs on its own thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.runs = {}

    def start(self, jobs, spec, drafts, output_root, input_root):
        identity = uuid.uuid4().hex[:12]
        state = dict(id=identity, phase='drafting', round=1, drafts=[], best=None, final=None, report=None, error=None)
        with self.lock:
            self.runs[identity] = state
        threading.Thread(target=self._run, args=(state, jobs, spec, drafts, output_root, input_root),
                         name='eclipse-drafts', daemon=True).start()
        return dict(state)

    def get(self, identity):
        with self.lock:
            state = self.runs.get(identity)
            return None if state is None else dict(state)

    def _set(self, state, **values):
        with self.lock:
            state.update(values)

    def _run(self, state, jobs, spec, drafts, output_root, input_root):
        try:
            run(jobs, spec, drafts, output_root, input_root, lambda **v: self._set(state, **v))
        except Exception as error:  # noqa: BLE001 - shown on the run, never raised into the server
            self._set(state, phase='failed', error=str(error) or type(error).__name__)


def wait_for(jobs, ids, wait=WAIT):
    """The finished rows of these jobs, once none is still active."""
    from ..agent.jobs import ACTIVE
    deadline = time.monotonic() + wait
    while True:
        rows = [jobs.get(i) for i in ids]
        if not any(row['status'] in ACTIVE for row in rows):
            return rows
        if time.monotonic() > deadline:
            raise TimeoutError('the drafts took longer than %d hours' % (wait // 3600))
        time.sleep(POLL)


def library_of(row):
    result = row.get('result') if row.get('status') == 'succeeded' else None
    return result.get('library_id') if isinstance(result, dict) else None


def run(jobs, spec, drafts, output_root, input_root, update=lambda **values: None):
    """Draft `drafts` seeds, rank them, retake once if none passes, finish the best. Returns the final library id."""
    from . import build, rank, score
    draft = {k: v for k, v in spec.items() if k not in ('takes', 'seed', 'preview', 'refine_from')}
    draft.update(kind='generate', preview=True)
    checked = []
    for number in range(1, ROUNDS + 1):
        update(phase='drafting', round=number)
        rows = jobs.submit(draft, takes=drafts)
        done = wait_for(jobs, [row['id'] for row in rows])
        made = [identity for identity in map(library_of, done) if identity]
        update(phase='checking')
        checked = sorted(checked + rank(output_root, input_root, made), key=score, reverse=True) if made else checked
        update(drafts=[dict(id=r['id'], passed=r['passed'], faces=[f.get('median') for f in r['faces']]) for r in checked])
        if checked and checked[0]['passed']:
            break
    if not checked:
        raise RuntimeError('no draft finished')
    best = checked[0]['id']
    update(phase='finishing', best=best)
    rows = jobs.submit(dict(kind='generate', refine_from=best))
    final = library_of(wait_for(jobs, [rows[0]['id']])[0])
    if not final:
        raise RuntimeError('the full-size take did not finish')
    report = build(output_root, input_root, final)
    update(phase='complete', final=final, report=report)
    return final
