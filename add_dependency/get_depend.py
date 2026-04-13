'''
从diff_code中提取依赖
version 20240628:
'''
from ast import Pass
import errno
import json
import re
import os
import shutil
import subprocess
import threading
import jsonlines
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import signal
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='tree_sitter')
from tree_sitter import Language, Parser


def _run_git(args, cwd, timeout=15):
    """Run a git command with hard process-group kill on timeout."""
    try:
        proc = subprocess.Popen(
            args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=_git_env, start_new_session=True
        )
        stdout, _ = proc.communicate(timeout=timeout)
        return stdout
    except subprocess.TimeoutExpired:
        # Kill the entire process group (includes lazy-fetch children)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()
        return None
    except Exception:
        return None

root = '/RepoSPD'

json_path =        root + '/dataset/example.json'
repo_path =        root + '/data_preproc/repo/'
cflow_path =       root + '/add_dependency/cflow_files/'
abfile_path =      root + '/data_preproc/ab_file/'
result_path =      root + '/dataset/example_dep.jsonl'
tree_sitter_path = root + '/add_dependency/libs/c-language.so'

_git_env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_LAZY_FETCH': '1'}
_ncpu = int(os.environ.get('CPU_LIMIT', 0)) or os.cpu_count() or 8


def _fetch_oids(repo_dir, oids):
    """Batch-fetch object IDs, splitting if arg list too long."""
    if not oids:
        return
    try:
        subprocess.run(
            ['git', 'fetch', 'origin'] + oids,
            cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_git_env, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print(f'  fetch timed out for {os.path.basename(repo_dir)}', flush=True)
    except OSError as e:
        if e.errno == errno.E2BIG and len(oids) > 1:
            mid = len(oids) // 2
            _fetch_oids(repo_dir, oids[:mid])
            _fetch_oids(repo_dir, oids[mid:])


_RECLONE_THRESHOLD = 5000  # repos with more .c blobs than this get re-cloned fully


def _prefetch_grep_blobs(data):
    """Ensure blobless repos have .c blobs local for git grep/show.

    Small repos: fetch individual .c blobs.
    Large repos (>5K .c files): re-clone fully (single packfile is faster).
    """
    repo_commits = defaultdict(set)
    for item in data:
        cid = item['commit_id']
        if cid:
            repo_commits[item['ori_dataset']].add(cid)

    # Find blobless repos
    blobless = {}
    for repo, commits in repo_commits.items():
        rp = repo_path + repo
        if not os.path.isdir(rp):
            continue
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.promisor'],
            cwd=rp, capture_output=True, text=True,
        )
        if result.stdout.strip() == 'true':
            blobless[repo] = commits

    if not blobless:
        return

    n = len(blobless)
    print(f'Ensuring .c blobs are local for {n} blobless repos...', flush=True)

    def _count_c_blobs(rp, rev):
        """Count .c blob OIDs at one revision (fast — only reads local tree objects)."""
        try:
            result = subprocess.run(
                ['git', 'ls-tree', '-r', rev],
                cwd=rp, capture_output=True, text=True,
                timeout=120, env=_git_env,
            )
            return sum(1 for line in result.stdout.splitlines()
                       if line.split('\t', 1)[-1].endswith('.c'))
        except (subprocess.TimeoutExpired, Exception):
            return 0

    def _reclone(repo):
        """Delete blobless clone and re-clone fully."""
        rp = repo_path + repo
        try:
            url = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=rp, capture_output=True, text=True,
            ).stdout.strip()
        except Exception:
            return False
        if not url:
            return False
        shutil.rmtree(rp, ignore_errors=True)
        result = subprocess.run(
            ['git', 'clone', '--quiet', url, repo],
            cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_git_env,
        )
        return result.returncode == 0

    def _prefetch_repo(repo, commits):
        rp = repo_path + repo

        # Check size at one commit to decide strategy
        sample_commit = next(iter(commits))
        c_count = _count_c_blobs(rp, sample_commit)

        if c_count > _RECLONE_THRESHOLD:
            print(f'  {repo}: {c_count} .c files — re-cloning fully...', flush=True)
            if _reclone(repo):
                print(f'  {repo}: re-cloned successfully', flush=True)
            else:
                print(f'  {repo}: re-clone failed', flush=True)
            return repo

        # Small repo: collect parent commits and fetch .c blobs
        all_revs = set(commits)
        for cid in commits:
            try:
                result = subprocess.run(
                    ['git', 'rev-list', '--parents', '-n', '1', cid],
                    cwd=rp, capture_output=True, text=True,
                    timeout=15, env=_git_env,
                )
                parts = result.stdout.strip().split()
                if len(parts) > 1:
                    all_revs.add(parts[1])
            except (subprocess.TimeoutExpired, Exception):
                pass

        # Baseline + diff-tree: ls-tree one rev, diff-tree the rest
        baseline = next(iter(all_revs))
        oids = set()
        try:
            result = subprocess.run(
                ['git', 'ls-tree', '-r', baseline],
                cwd=rp, capture_output=True, text=True,
                timeout=120, env=_git_env,
            )
            for line in result.stdout.splitlines():
                parts = line.split(None, 3)
                if len(parts) >= 4 and parts[3].endswith('.c'):
                    oids.add(parts[2])
        except (subprocess.TimeoutExpired, Exception):
            pass

        # For other revisions, only fetch changed .c blobs via diff-tree
        null_oid = '0' * 40
        for rev in all_revs:
            if rev == baseline:
                continue
            try:
                result = subprocess.run(
                    ['git', 'diff-tree', '-r', baseline, rev],
                    cwd=rp, capture_output=True, text=True,
                    timeout=60, env=_git_env,
                )
                for line in result.stdout.splitlines():
                    if not line.startswith(':'):
                        continue
                    tab = line.find('\t')
                    if tab < 0 or not line[tab+1:].endswith('.c'):
                        continue
                    meta = line[:tab].split()
                    if len(meta) >= 4:
                        for oid in (meta[2], meta[3]):
                            if oid != null_oid:
                                oids.add(oid)
            except (subprocess.TimeoutExpired, Exception):
                pass

        if oids:
            print(f'  {repo}: fetching {len(oids)} .c blobs ({len(all_revs)} revisions)...', flush=True)
            _fetch_oids(rp, list(oids))

        return repo

    done = [0]
    with ThreadPoolExecutor(max_workers=min(32, _ncpu)) as pool:
        futures = {pool.submit(_prefetch_repo, repo, commits): repo
                   for repo, commits in blobless.items()}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f'  {futures[f]}: error ({e})', flush=True)
            done[0] += 1
            print(f'[{done[0]}/{n}] repos prefetched', flush=True)


# Thread-local cache for tree-sitter parser (avoid reloading the .so per item)
_tls = threading.local()

def _get_parser():
    if not hasattr(_tls, 'parser'):
        lang = Language(tree_sitter_path, 'c')
        _tls.parser = Parser()
        _tls.parser.set_language(lang)
    return _tls.parser


def load_cflow(path):
    a_file = ''
    b_file = ''
    with open(path,'r') as f:
        a_file = f.read()
    a_def = re.findall(r'[\\+]-(\w+)\(\) \<', a_file)
    a_funcs = re.findall(r'[\\+]-(\w+)\(', a_file)
    a_funcs = list(set(a_funcs)-set(a_def))
    with open(path.replace('_a.cflow','_b.cflow'),'r') as f:
        b_file = f.read()
    b_def = re.findall(r'[\\+]-(\w+)\(\) \<', b_file)
    b_funcs = re.findall(r'[\\+]-(\w+)\(', b_file)
    b_funcs = list(set(b_funcs)-set(b_def))
    diff = [i for i in a_funcs if i not in b_funcs] + [i for i in b_funcs if i not in a_funcs]
    pre_diff = [i for i in a_funcs if i not in b_funcs]
    post_diff = [i for i in b_funcs if i not in a_funcs]
    return diff, pre_diff, post_diff


def find_func_in_repo(funcs, repo_name, commit_id, preFuncs, postFuncs):
    """Find function definitions in a repo using git grep + git show.

    Read-only git operations — safe to call concurrently without locking.
    """
    repo_dir = repo_path + repo_name
    pre_funcDef = {}
    post_funcDef = {}
    find_funcs = {}

    parser = _get_parser()

    def _search_at_commit(target_commit, func_names, func_dict):
        if not func_names:
            return
        remaining = set(func_names)

        # Use git grep to find only .c files that mention any of the names
        grep_args = []
        for name in func_names:
            grep_args.extend(['-e', name])
        stdout = _run_git(
            ['git', 'grep', '-l'] + grep_args + [target_commit, '--', '*.c'],
            cwd=repo_dir, timeout=10
        )
        if not stdout:
            return
        grep_output = stdout.decode('utf-8', errors='ignore').strip()
        if not grep_output:
            return

        # Collect file paths to fetch
        file_paths = []
        for line in grep_output.split('\n'):
            if ':' not in line:
                continue
            file_paths.append(line.split(':', 1)[1])

        if not file_paths:
            return

        # Use git cat-file --batch to read all blobs in one process
        refs = ''.join(f'{target_commit}:{fp}\n' for fp in file_paths)
        try:
            proc = subprocess.Popen(
                ['git', 'cat-file', '--batch'],
                cwd=repo_dir, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=_git_env, start_new_session=True
            )
            batch_out, _ = proc.communicate(input=refs.encode(), timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return
        except Exception:
            return

        # Parse batch output: each blob is "<sha> blob <size>\n<content>\n"
        pos = 0
        for fp in file_paths:
            if not remaining:
                break
            # Read header line
            nl = batch_out.find(b'\n', pos)
            if nl < 0:
                break
            header = batch_out[pos:nl].decode('utf-8', errors='ignore')
            pos = nl + 1
            parts = header.split()
            if len(parts) < 3 or parts[1] == 'missing':
                continue
            try:
                blob_size = int(parts[2])
            except ValueError:
                continue
            blob = batch_out[pos:pos + blob_size]
            pos += blob_size + 1  # skip trailing \n
            try:
                code = blob.decode('utf-8')
            except UnicodeDecodeError:
                continue
            try:
                tree = parser.parse(bytes(code, 'utf-8'))
                for node in tree.root_node.children:
                    if node.type == 'function_definition':
                        for child in node.children:
                            if child.type == 'function_declarator':
                                fname = child.children[0].text.decode('utf-8').strip()
                                if fname in remaining:
                                    find_funcs[fname] = node.text.decode('utf-8')
                                    func_dict[fname] = node.text.decode('utf-8')
                                    remaining.discard(fname)
            except (UnicodeDecodeError, Exception):
                continue

    # Post-patch: search at commit_id
    _search_at_commit(commit_id, postFuncs, post_funcDef)

    # Get parent commit for pre-patch
    stdout = _run_git(
        ['git', 'rev-list', '--parents', '-n', '1', commit_id],
        cwd=repo_dir, timeout=15
    )
    if stdout:
        out = stdout.decode('utf-8', errors='ignore').strip()
        pre_patch = out[out.find(' ')+1:].rstrip()
    else:
        pre_patch = ''

    # Pre-patch: search at parent commit
    if pre_patch:
        _search_at_commit(pre_patch, preFuncs, pre_funcDef)

    for func in funcs:
        if func not in find_funcs:
            find_funcs[func] = ''
    return find_funcs, pre_funcDef, post_funcDef


def _process_item(item):
    find_funcs = {}
    pre_findFuncs = {}
    post_findFuncs = {}
    repo = item['ori_dataset']
    commit_id = item['commit_id']
    if commit_id == '':
        return None

    path_a = abfile_path + commit_id + '/a/'
    path_b = abfile_path + commit_id + '/b/'
    cf_dir = cflow_path + commit_id
    os.makedirs(cf_dir, exist_ok=True)

    diff_funcs = []
    pre_diffFuncs = []
    post_diffFuncs = []

    if os.path.exists(path_a):
        a_files = [f for f in os.listdir(path_a) if f.endswith('.c')]
        has_b = os.path.exists(path_b)
        for f in a_files:
            afile = path_a + f
            a_cflow = f'{cf_dir}/{f}_a.cflow'
            b_cflow = f'{cf_dir}/{f}_b.cflow'

            if not os.path.exists(a_cflow):
                try:
                    with open(a_cflow, 'w') as out:
                        subprocess.run(
                            ['cflow', '-T', afile],
                            stdout=out, stderr=subprocess.DEVNULL, timeout=10
                        )
                except (subprocess.TimeoutExpired, Exception):
                    pass
            if has_b:
                bfile = afile.replace('/a/','/b/')
                if not os.path.exists(b_cflow):
                    try:
                        with open(b_cflow, 'w') as out:
                            subprocess.run(
                                ['cflow', '-T', bfile],
                                stdout=out, stderr=subprocess.DEVNULL, timeout=10
                            )
                    except (subprocess.TimeoutExpired, Exception):
                        pass
                try:
                    tmp_diff, tmp_pre, tmp_post = load_cflow(a_cflow)
                    diff_funcs.extend(tmp_diff)
                    pre_diffFuncs.extend(tmp_pre)
                    post_diffFuncs.extend(tmp_post)
                    diff_funcs = list(set(diff_funcs))
                    pre_diffFuncs = list(set(pre_diffFuncs))
                    post_diffFuncs = list(set(post_diffFuncs))
                except Exception:
                    pass

    if os.path.exists(repo_path+repo) and diff_funcs:
        find_funcs, pre_findFuncs, post_findFuncs = find_func_in_repo(
            diff_funcs, repo, commit_id, pre_diffFuncs, post_diffFuncs)

    item['dependency'] = find_funcs
    item['pre_dep'] = pre_findFuncs
    item['post_dep'] = post_findFuncs
    return item


def main():
    os.makedirs(cflow_path, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    _prefetch_grep_blobs(data)

    data_finish = []
    if os.path.isfile(result_path):
        with open(result_path, 'r') as rf:
            lines = rf.readlines()
        good_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data_finish.append(json.loads(line))
                good_lines.append(line)
            except json.JSONDecodeError:
                pass
        # Rewrite without corrupt lines
        if len(good_lines) < len([l for l in lines if l.strip()]):
            with open(result_path, 'w') as wf:
                for gl in good_lines:
                    wf.write(gl + '\n')
    id_finish = set(item['commit_id'] for item in data_finish)

    todo = [item for item in data if item['commit_id'] and item['commit_id'] not in id_finish]
    total = len(todo)
    print(f'{total} items to process ({len(id_finish)} already done)', flush=True)

    if not todo:
        return

    # Pass 1: fast-path items that have no ab_file (no cflow/git work needed)
    need_work = []
    fast = 0
    with open(result_path, 'a', buffering=8192) as fout:
        for item in todo:
            cid = item['commit_id']
            path_a = abfile_path + cid + '/a/'
            if not os.path.exists(path_a):
                # No source files — write empty deps immediately
                item['dependency'] = {}
                item['pre_dep'] = {}
                item['post_dep'] = {}
                fout.write(json.dumps(item) + '\n')
                fast += 1
            else:
                need_work.append(item)
        fout.flush()
    print(f'Pass 1: {fast} items with no source files (instant)', flush=True)
    print(f'Pass 2: {len(need_work)} items need cflow/git processing', flush=True)

    if not need_work:
        return

    # Interleave by repo so workers spread across repos
    from collections import defaultdict as _dd
    by_repo = _dd(list)
    no_repo = []
    for item in need_work:
        repo = item.get('ori_dataset', '')
        rp = repo_path + repo
        if repo and os.path.isdir(rp):
            by_repo[repo].append(item)
        else:
            no_repo.append(item)
    interleaved = []
    repo_iters = [iter(items) for items in by_repo.values()]
    while repo_iters:
        next_round = []
        for it in repo_iters:
            item = next(it, None)
            if item is not None:
                interleaved.append(item)
                next_round.append(it)
        repo_iters = next_round
    interleaved.extend(no_repo)

    done = 0
    total_work = len(interleaved)
    _nworkers = min(128, _ncpu)
    print(f'Starting {_nworkers} process workers ({len(by_repo)} repos)...', flush=True)

    # Use ProcessPoolExecutor to bypass the GIL entirely.
    # Each worker process gets its own GIL — tree-sitter, regex, and
    # subprocess calls all run truly in parallel.
    # id_finish is passed per-item to avoid pickling issues with large sets.
    with open(result_path, 'a', buffering=8192) as fout:
        with ProcessPoolExecutor(max_workers=_nworkers) as pool:
            futures = {pool.submit(_process_item, item): item for item in interleaved}
            for f in as_completed(futures):
                try:
                    result = f.result()
                except Exception as e:
                    cid = futures[f].get('commit_id', '?')
                    print(f'  error {cid}: {e}', flush=True)
                    result = None
                if result is not None:
                    fout.write(json.dumps(result) + '\n')
                done += 1
                if done <= 10 or done % 500 == 0 or done == total_work:
                    fout.flush()
                    print(f'[{done}/{total_work}] dependencies extracted', flush=True)

if __name__ == '__main__':
    main()
