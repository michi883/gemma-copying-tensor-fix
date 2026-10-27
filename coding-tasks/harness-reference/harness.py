#!/usr/bin/env python3
"""One coding-agent trial on one gemma-edit-tasks repository, against one OpenAI-compatible model server.

    python harness.py TASK_ID --arm NAME --url http://127.0.0.1:8110/v1 --model MODEL --out RUN_DIR [--work-root DIR]

What is reused unchanged from the real swegemma Phase 1 harness (swegemma 0.2.11, ~/gemma4-shell/.venv):
  - the tool implementations read_file / edit_file / write_file / run_command / submit_patch / get_status (swegemma.tools), including their JSON responses, budget gating and the
    edit matcher (adk_eval_core apply_replacement: exact -> flexible -> regex),
  - build_agent_prompt + build_workspace_tree_command (the user prompt, budgets, execution rules, workspace tree),
  - the tool declarations and the system message, copied byte for byte from a recorded request of the lab (tools.json, system message below).
What is new: the sandbox is a local directory (LocalSandbox maps /workspace to a fresh copy of the task repository, git-initialised with a baseline commit), and the model loop is ~60 lines
of OpenAI chat-completions calls.  Settings that differ from the lab's baseline run on purpose: thinking OFF (chat_template_kwargs.enable_thinking=false, no thinking budget), temperature 0,
fixed seed, no context compaction, no continuation nudges (a text-only reply ends the session, as the prompt's instruction 5 says).  Identical for every arm.

Nothing about the model, its tokenizer or the copying defect is mentioned anywhere; the task prompt is the repository's ISSUE.md wrapped by build_agent_prompt.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TASKS = HERE.parent / 'tasks'
ENV_BIN = HERE.parent.parent / 'envs' / 'requests' / 'bin'          # python 3.12 + pytest, used for the sandbox's `python` / `pytest` and for grading
SHIMS = HERE / 'sandbox_bin'                                            # pytest / py.test -> `python -m pytest` (the env's own console scripts have a stale shebang)
SYSTEM_MESSAGE = 'You are an agent. Your internal name is "swe_agent".'      # as recorded in the lab's requests (swegemma's agent wrapper)
BUDGET = dict(time_minutes=8.0, tool_calls=28, turns=80)               # the lab's eval_config.yaml
LIMITS = dict(command_timeout_seconds=240, max_stdout_chars=5000, max_file_lines=150, max_file_chars=10000)
GEN = dict(temperature=0, top_p=0.95, max_completion_tokens=8192, seed=1)
EXTRA_BODY = {'chat_template_kwargs': {'enable_thinking': False}, 'cache_prompt': False}      # llama-server reads these at the top level of the body (a nested extra_body is ignored)


class LocalSandbox:
    """The few methods of swegemma's sandbox manager that the tools call, on a local directory standing in for /workspace."""

    def __init__(self, workdir: Path):
        self.root = str(workdir.resolve())

    def _map(self, text: str) -> str:
        return text.replace('/workspace', self.root)

    def _unmap(self, text: str) -> str:
        return text.replace(self.root, '/workspace')

    def exec(self, container_id, command, timeout=None):
        env = {'PATH': f'{SHIMS}:{ENV_BIN}:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin', 'SANDBOX_PYTHON': str(ENV_BIN / 'python'), 'HOME': self.root, 'LANG': 'en_US.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTEST_ADDOPTS': '-p no:cacheprovider', 'PYTHONNOUSERSITE': '1', 'GIT_AUTHOR_NAME': 'agent', 'GIT_AUTHOR_EMAIL': 'agent@localhost',
               'GIT_COMMITTER_NAME': 'agent', 'GIT_COMMITTER_EMAIL': 'agent@localhost'}
        try:
            p = subprocess.run(['/bin/bash', '-c', self._map(command)], cwd=self.root, env=env, capture_output=True, text=True, timeout=timeout, errors='replace')
            return SimpleNamespace(stdout=self._unmap(p.stdout), stderr=self._unmap(p.stderr), exit_code=p.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            return SimpleNamespace(stdout='', stderr='', exit_code=-1, timed_out=True)

    def copy_from(self, container_id, src, local_path):
        s = Path(self._map(str(src)))
        if s.exists():
            shutil.copyfile(s, local_path)

    def copy_to(self, container_id, local_path, dest):
        d = Path(self._map(str(dest)))
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, d / Path(local_path).name)


def sh(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def prepare_workdir(task_id: str, workdir: Path):
    """Fresh copy of the pristine task directory + a one-commit git repository (the baseline `submit_patch` diffs against)."""
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(TASKS / task_id, workdir, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.DS_Store'))
    g = ['git', '-c', 'user.name=baseline', '-c', 'user.email=baseline@localhost']
    sh(g + ['init', '-q', '-b', 'main'], workdir, check=True)
    (workdir / '.git' / 'info' / 'exclude').write_text('__pycache__/\n.pytest_cache/\n*.pyc\n')
    sh(g + ['add', '-A'], workdir, check=True)
    sh(g + ['commit', '-q', '-m', 'baseline'], workdir, check=True)
    sh(['git', 'tag', '_swegemma_baseline'], workdir, check=True)


def tree_hash(root: Path):
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
            h.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes() + b'\0')
    return h.hexdigest()


def chat(url, model, messages, tools):
    body = {'model': model, 'messages': messages, 'tools': tools, **GEN, **EXTRA_BODY}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url.rstrip('/') + '/chat/completions', data=data, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return body, resp, round(time.time() - t0, 2)


def grade(workdir: Path, task_id: str):
    """Restore the pristine test files (the scorer discards edits to tests), then run the task's tests exactly as the README says: python -m pytest -q."""
    modified_tests = []
    for f in sorted(p for p in (TASKS / task_id).iterdir() if p.name.startswith('test_') and p.suffix == '.py'):
        if (workdir / f.name).read_bytes() != f.read_bytes():
            modified_tests.append(f.name)
            shutil.copyfile(f, workdir / f.name)
    env = {'PATH': f'{ENV_BIN}:/usr/bin:/bin', 'HOME': str(workdir), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
    p = subprocess.run([str(ENV_BIN / 'python'), '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], cwd=workdir, env=env, capture_output=True, text=True, timeout=120)
    tail = (p.stdout + p.stderr).strip().splitlines()[-1] if (p.stdout + p.stderr).strip() else ''
    return {'passed': p.returncode == 0, 'exit_code': p.returncode, 'summary': tail, 'output': (p.stdout + p.stderr)[-2000:], 'test_files_modified_by_agent': modified_tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task')
    ap.add_argument('--arm', required=True)
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work-root', default=os.environ.get('TMPDIR', '/tmp'))
    ap.add_argument('--extra-body', default=None, help='JSON object replacing the default extra request fields (llama-server: chat_template_kwargs + cache_prompt); e.g. for vLLM: \'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    a = ap.parse_args()
    if a.extra_body:
        EXTRA_BODY.clear()
        EXTRA_BODY.update(json.loads(a.extra_body))

    from swegemma.budget import EvaluationBudget, HarnessLimits
    from swegemma.context import SwegemmaContext
    from swegemma.harness.agent_runner import build_agent_prompt, build_workspace_tree_command
    from swegemma.models import Task
    from swegemma.tools import create_tools

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(a.work_root) / f'{a.arm}_{a.task}'
    prepare_workdir(a.task, workdir)
    start_hash = tree_hash(workdir)
    assert start_hash == tree_hash(TASKS / a.task), 'working copy differs from the pristine task directory'
    sandbox = LocalSandbox(workdir)

    meta = {t['id']: t for t in json.loads((HERE.parent / 'tasks.json').read_text())}[a.task]
    task = Task(instance_id=a.task, repo=f'gemma-edit-tasks/{a.task}', base_commit='baseline', problem_statement=(TASKS / a.task / 'ISSUE.md').read_text().rstrip('\n'), hints_text='', created_at='')
    budget, harness = EvaluationBudget(**BUDGET), HarnessLimits(**LIMITS)
    ctx = SwegemmaContext(docker_manager=sandbox, container_id='local', task=task, budget=budget, harness=harness)
    ctx.start_agent_session()
    tools = create_tools(ctx)
    tool_decls = json.loads((HERE / 'tools.json').read_text())
    declared = [t['function']['name'] for t in tool_decls]

    tree = sandbox.exec('local', build_workspace_tree_command(), timeout=30).stdout.strip()
    prompt = build_agent_prompt(task, SimpleNamespace(budget=budget, harness=harness), tree, declared_tools=declared)
    messages = [{'role': 'system', 'content': SYSTEM_MESSAGE}, {'role': 'user', 'content': prompt}]

    events = open(out / 'events.jsonl', 'w')

    def emit(kind, **kw):
        events.write(json.dumps({'t': round(time.time() - t_start, 2), 'kind': kind, **kw}, ensure_ascii=False) + '\n')
        events.flush()

    t_start = time.time()
    emit('run_start', arm=a.arm, task=a.task, url=a.url, model=a.model, gen=GEN, extra_body=EXTRA_BODY, budget=BUDGET, limits=LIMITS, start_tree_sha256=start_hash)
    end_reason, submitted, n_turns = 'text_only_reply', False, 0
    for turn in range(1, BUDGET['turns'] + 1):
        if time.time() - t_start > BUDGET['time_minutes'] * 60:
            end_reason = 'time_budget'
            break
        body, resp, secs = chat(a.url, a.model, messages, tool_decls)
        ctx.record_llm_call()
        n_turns = turn
        ch = resp['choices'][0]
        msg = ch['message']
        emit('model_reply', turn=turn, seconds=secs, finish_reason=ch['finish_reason'], usage=resp.get('usage'), message=msg, request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
             served_model=resp.get('model'), reasoning_chars=len(msg.get('reasoning_content') or ''))
        calls = msg.get('tool_calls') or []
        assistant = {'role': 'assistant', 'content': msg.get('content') or ''}
        if calls:
            assistant['tool_calls'] = [{'type': 'function', 'id': c['id'], 'function': {'name': c['function']['name'], 'arguments': c['function']['arguments']}} for c in calls]
        messages.append(assistant)
        if not calls:
            end_reason = 'text_only_reply' if ch['finish_reason'] == 'stop' else f'finish_reason={ch["finish_reason"]}'
            break
        for c in calls:
            name, raw = c['function']['name'], c['function']['arguments']
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if name not in tools:
                    result = {'status': 'error', 'error_type': 'UnknownTool', 'error_message': f'no tool named {name}'}
                else:
                    result = tools[name](**args)
            except Exception as e:                                   # bad arguments: the lab's ADK layer would also return an error to the model
                args = raw
                result = {'status': 'error', 'error_type': type(e).__name__, 'error_message': str(e)}
            if name == 'submit_patch' and isinstance(result, dict) and result.get('status') == 'ok':
                submitted = True
            emit('tool_call', turn=turn, name=name, args=args, raw_arguments=raw)
            emit('tool_result', turn=turn, name=name, result=result)
            messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': json.dumps(result, ensure_ascii=False)})
            if isinstance(result, dict) and result.get('error_type') == 'BudgetExceeded':      # "Session terminates when any budget is exhausted" (the prompt); every state-changing tool is gated, so the files cannot change afterwards
                end_reason = 'tool_call_budget'
                break
        if end_reason == 'tool_call_budget':
            break
    else:
        end_reason = 'turn_budget'

    sandbox.exec('local', 'cd /workspace && git add -N .', timeout=30)
    patch = sandbox.exec('local', 'cd /workspace && git diff --binary HEAD', timeout=30).stdout
    graded = grade(workdir, a.task)
    after = {}
    for p in sorted(workdir.iterdir()):
        if p.is_file() and p.suffix == '.py':
            after[p.name] = p.read_text()
    result = {'arm': a.arm, 'task': a.task, 'target_identifier': meta['target_identifier'], 'end_reason': end_reason, 'submit_patch_called': submitted, 'turns': n_turns,
              'tool_calls_used': ctx.tool_calls_used, 'seconds': round(time.time() - t_start, 1), 'tests': graded, 'final_patch': patch, 'final_files': after}
    emit('run_end', end_reason=end_reason, tests_passed=graded['passed'], tool_calls_used=ctx.tool_calls_used)
    (out / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / 'transcript.json').write_text(json.dumps(messages, indent=1, ensure_ascii=False))
    (out / 'final.patch').write_text(patch)
    shutil.rmtree(workdir, ignore_errors=True)
    print(json.dumps({k: result[k] for k in ('arm', 'task', 'end_reason', 'submit_patch_called', 'turns', 'tool_calls_used', 'seconds')} | {'tests': graded['summary']}))


if __name__ == '__main__':
    main()
