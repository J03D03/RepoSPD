import errno
import io
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from locate_and_align import locate_and_align
from merge_cpg import getdirsize, generateLog

logger = logging.getLogger(__name__)

root = '/RepoSPD' # Absolute path to `release`

path =            root + '/dataset/example.json'
path_ab_file =    root + '/data_preproc/ab_file/' # path of ab_file
path_repo =       root + '/data_preproc/repo/' # path of repositories
path_joern =      root + '/joern/'
path_locateFunc = root + '/data_preproc/locateFunc.sc' #script file
path_locateFunc_batch = root + '/data_preproc/locateFunc_batch.sc'

_ncpu = int(os.environ.get('CPU_LIMIT', 0)) or os.cpu_count() or 8

_JOERN_HEAP = os.environ.get('JOERN_HEAP', '4g')
_JOERN_TIMEOUT = int(os.environ.get('JOERN_TIMEOUT', '300'))
_JOERN_JVM_OPTS = [f'-J-Xmx{_JOERN_HEAP}', '-J-XX:TieredStopAtLevel=1', '-J-noverify']

# get_ab_file

FULL_CLONE_THRESHOLD = 0  # always full-clone (blobless causes repeated prefetch overhead)


def _convert_blobless_to_full(workers=8):
    """Convert any blobless (partial) clones to full clones."""
    if not os.path.isdir(path_repo):
        return
    blobless = []
    for name in os.listdir(path_repo):
        repo_dir = os.path.join(path_repo, name)
        if not os.path.isdir(repo_dir):
            continue
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.promisor'],
            cwd=repo_dir, capture_output=True, text=True,
        )
        if result.stdout.strip() == 'true':
            blobless.append(name)

    if not blobless:
        return

    print(f'Converting {len(blobless)} blobless repos to full clones...', flush=True)

    def _convert_one(name):
        repo_dir = os.path.join(path_repo, name)
        subprocess.run(
            ['git', 'fetch', '--unshallow'],
            cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_git_env,
        )
        # If --unshallow fails (not shallow), do a regular fetch
        subprocess.run(
            ['git', 'fetch', 'origin'],
            cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_git_env,
        )
        subprocess.run(
            ['git', 'config', '--unset', 'remote.origin.promisor'],
            cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ['git', 'config', '--unset', 'remote.origin.partialclonefilter'],
            cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return name

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_convert_one, n): n for n in blobless}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.debug('convert error %s: %s', futures[f], e)
            done += 1
            print(f'[{done}/{len(blobless)}] repos converted ({futures[f]})', flush=True)


def clone_repos(workers=min(8, _ncpu)):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if not os.path.exists(path_repo):
        os.mkdir(path_repo)

    # remove broken repos left by previous interrupted runs
    for name in os.listdir(path_repo):
        repo_dir = os.path.join(path_repo, name)
        if os.path.isdir(repo_dir) and subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode != 0:
            logger.debug('Removing broken repo: %s', name)
            shutil.rmtree(repo_dir)

    # convert any remaining blobless clones to full clones
    _convert_blobless_to_full(workers)

    # count items per repo to decide clone strategy
    item_counts = Counter(item['ori_dataset'] for item in data if item['commit_id'])

    # collect unique repos that need cloning
    repos = {}
    for item in data:
        repo = item['ori_dataset']
        if repo not in repos and not os.path.exists(path_repo + repo):
            repos[repo] = item['project_url']

    n_full = sum(1 for r in repos if item_counts[r] >= FULL_CLONE_THRESHOLD)
    print(f'Found {len(repos)} repos to clone ({n_full} full, {len(repos) - n_full} blobless)', flush=True)

    def _clone_one(repo_url):
        repo, url = repo_url
        env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
        cmd = ['git', 'clone', '--quiet', url, repo]
        if item_counts[repo] < FULL_CLONE_THRESHOLD:
            cmd.insert(2, '--filter=blob:none')
        subprocess.run(
            cmd,
            cwd=path_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env,
        )
        return repo

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_clone_one, item): item for item in repos.items()}
        for f in as_completed(futures):
            repo = f.result()
            done += 1
            print(f'[{done}/{len(repos)}] repos cloned ({repo})', flush=True)



def _parse_diff_paths(diff_code):
    a = []
    b = []
    i = diff_code.find("diff --git")
    content = diff_code[i:]
    while 'diff --git ' in content:
        i = content.find(' a/')
        j = content.find(' b/')
        k = content.find('\n')
        file_a = content[i + 3:]
        i = file_a.find(' ')
        file_a = file_a[:i]
        file_b = content[j + 3:k]
        if file_a not in a:
            a.append(file_a)
        if file_b not in b:
            b.append(file_b)
        i = content.find('\ndiff --git ')
        if i > 0:
            content = content[i + 1:]
        else:
            break
    return a, b


_git_env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}

def _fetch_oids(repo_path, oids, deadline=None):
    if not oids:
        return
    if deadline is None:
        deadline = time.monotonic() + 1800
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    try:
        subprocess.run(
            ['git', 'fetch', 'origin'] + oids,
            cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_git_env, timeout=remaining,
        )
    except subprocess.TimeoutExpired:
        repo = os.path.basename(repo_path)
        logger.debug('%s: fetch timed out (%d oids), skipping', repo, len(oids))
    except OSError as e:
        if e.errno == errno.E2BIG and len(oids) > 1:
            mid = len(oids) // 2
            _fetch_oids(repo_path, oids[:mid], deadline)
            _fetch_oids(repo_path, oids[mid:], deadline)
        else:
            raise


def _prefetch_one(repo, commit_ids, timeout=1200):
    repo_path = path_repo + repo
    if not os.path.exists(repo_path):
        return repo

    deadline = time.monotonic() + timeout

    def _remaining():
        r = deadline - time.monotonic()
        if r <= 0:
            raise TimeoutError(f'{repo}: overall deadline exceeded')
        return r

    # skip repos that were fully cloned (all blobs already local)
    result = subprocess.run(
        ['git', 'config', '--get', 'remote.origin.promisor'],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.stdout.strip() != 'true':
        return repo

    logger.debug('%s: starting (%d commits)...', repo, len(commit_ids))
    proc = None
    try:
        # Single diff-tree call replaces git log + N × git ls-tree.
        # Stream output line-by-line to avoid buffering the full result.
        proc = subprocess.Popen(
            ['git', 'diff-tree', '-r', '-m', '--stdin', '--no-commit-id'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=repo_path, text=True,
        )
        def _write_stdin():
            try:
                proc.stdin.write('\n'.join(commit_ids))
            except OSError:
                pass
            finally:
                proc.stdin.close()
        writer = threading.Thread(target=_write_stdin, daemon=True)
        writer.start()

        null_oid = '0' * 40
        oids = set()
        for line in proc.stdout:
            _remaining()
            if not line.startswith(':'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                for oid in (parts[2], parts[3]):
                    if oid != null_oid:
                        oids.add(oid)

        if oids:
            logger.debug('%s: fetching %d blobs...', repo, len(oids))
            _fetch_oids(repo_path, list(oids), deadline)
    except (subprocess.TimeoutExpired, TimeoutError) as e:
        logger.debug('%s: timed out (%s), skipping', repo, e)
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()
    return repo


def prefetch_blobs(workers=min(32, _ncpu)):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    repo_commits = defaultdict(set)
    for item in data:
        if item['commit_id']:
            repo_commits[item['ori_dataset']].add(item['commit_id'])
    del data

    total = len(repo_commits)
    print(f'Prefetching blobs for {total} repos', flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_prefetch_one, repo, list(commits)): repo
                   for repo, commits in repo_commits.items()}
        del repo_commits
        for f in as_completed(futures):
            f.result()
            done += 1
            print(f'[{done}/{total}] prefetched ({futures[f]})', flush=True)


def _subprocess_run_retry(*args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return subprocess.run(*args, **kwargs)
        except BlockingIOError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def _archive_files(repo_path, commit, files, dest_dir, timeout=60):
    """Extract multiple files from a commit using git archive (single subprocess)."""
    if not files:
        return
    try:
        proc = _subprocess_run_retry(
            ['git', 'archive', '--format=tar', commit, '--'] + files,
            cwd=repo_path, capture_output=True, timeout=timeout,
            env=_git_env,
        )
    except subprocess.TimeoutExpired:
        return
    if proc.returncode != 0 or not proc.stdout:
        # fallback: try files individually (some may have been deleted/renamed)
        for f in files:
            try:
                result = _subprocess_run_retry(
                    ['git', 'show', f'{commit}:{f}'],
                    cwd=repo_path, capture_output=True, timeout=30,
                    env=_git_env,
                )
                if result.returncode == 0:
                    dest = os.path.join(dest_dir, os.path.basename(f))
                    with open(dest, 'wb') as out:
                        out.write(result.stdout)
            except subprocess.TimeoutExpired:
                continue
        return
    # unpack tar from memory
    try:
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                dest = os.path.join(dest_dir, os.path.basename(member.name))
                with tar.extractfile(member) as src, open(dest, 'wb') as out:
                    out.write(src.read())
    except (tarfile.TarError, EOFError):
        pass


def _extract_ab(item):
    commit_id = item['commit_id']
    if commit_id == '':
        return
    repo = item['ori_dataset']

    dir_a = path_ab_file + commit_id + '/a'
    dir_b = path_ab_file + commit_id + '/b'

    # skip if already extracted (both sides must have files)
    if os.path.isdir(dir_a) and os.path.isdir(dir_b):
        if os.listdir(dir_a) and os.listdir(dir_b):
            return

    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    a, b = _parse_diff_paths(item['diff_code'])

    repo_path = path_repo + repo

    if not os.path.isdir(repo_path):
        return

    # get parent commit for pre-patch files
    try:
        out = _subprocess_run_retry(
            ['git', 'rev-list', '--parents', '-n', '1', commit_id],
            cwd=repo_path, capture_output=True, text=True,
            timeout=30, env=_git_env,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        return
    commit_a = out[out.find(' ') + 1:]

    # extract post-patch files (single archive call)
    _archive_files(repo_path, commit_id, b, dir_b)

    # extract pre-patch files (single archive call)
    if commit_a:
        _archive_files(repo_path, commit_a, a, dir_a)


def _batch_prefetch(repo_path, commits):
    """Batch-fetch all blobs needed for the given commits in a blobless repo."""
    result = subprocess.run(
        ['git', 'config', '--get', 'remote.origin.promisor'],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.stdout.strip() != 'true':
        return  # fully cloned, all blobs already local

    proc = None
    try:
        proc = subprocess.Popen(
            ['git', 'diff-tree', '-r', '-m', '--stdin', '--no-commit-id'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=repo_path, text=True,
        )
        stdout, _ = proc.communicate(input='\n'.join(commits), timeout=300)
    except (subprocess.TimeoutExpired, OSError):
        if proc is not None:
            proc.kill()
            proc.wait()
        return

    null_oid = '0' * 40
    oids = set()
    for line in stdout.split('\n'):
        if not line.startswith(':'):
            continue
        parts = line.split()
        if len(parts) >= 5:
            for oid in (parts[2], parts[3]):
                if oid != null_oid:
                    oids.add(oid)

    if oids:
        repo = os.path.basename(repo_path)
        print(f'  {repo}: fetching {len(oids)} blobs...', flush=True)
        _fetch_oids(repo_path, list(oids))


def get_ab_file(workers=_ncpu):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    os.makedirs(path_ab_file, exist_ok=True)

    # Group by repo to avoid git lock contention and enable batch blob prefetch
    repo_items = defaultdict(list)
    for item in data:
        if item['commit_id']:
            repo_items[item['ori_dataset']].append(item)

    total = sum(len(v) for v in repo_items.values())

    # Count already-extracted items upfront (single-threaded, fast)
    already_done = 0
    no_repo = 0
    repos_to_process = {}
    for repo, items in repo_items.items():
        repo_path = path_repo + repo
        if not os.path.isdir(repo_path):
            no_repo += len(items)
            continue
        todo = []
        for item in items:
            cid = item['commit_id']
            dir_a = path_ab_file + cid + '/a'
            dir_b = path_ab_file + cid + '/b'
            if os.path.isdir(dir_a) and os.path.isdir(dir_b) \
               and os.listdir(dir_a) and os.listdir(dir_b):
                already_done += 1
            else:
                todo.append(item)
        if todo:
            repos_to_process[repo] = todo

    done_count = [already_done + no_repo]
    done_lock = threading.Lock()
    remaining = total - already_done - no_repo
    print(f'{already_done}/{total} already extracted, {remaining} to do across {len(repos_to_process)} repos'
          + (f' ({no_repo} skipped — repo not cloned)' if no_repo else ''),
          flush=True)

    if not repos_to_process:
        return

    def _extract_and_count(item):
        _extract_ab(item)
        with done_lock:
            done_count[0] += 1
            if done_count[0] <= 10 or done_count[0] % 100 == 0 \
               or done_count[0] >= total - 5:
                print(f'[{done_count[0]}/{total}] ab files extracted', flush=True)

    # Phase 1: prefetch blobs per repo (I/O bound, parallelise across repos)
    n_repos = len(repos_to_process)
    print(f'Prefetching blobs for {n_repos} repos...', flush=True)
    prefetch_done = [0]
    with ThreadPoolExecutor(max_workers=min(32, workers)) as prefetch_pool:
        prefetch_futures = {
            prefetch_pool.submit(_batch_prefetch, path_repo + repo,
                                 [item['commit_id'] for item in todo]): repo
            for repo, todo in repos_to_process.items()
        }
        for f in as_completed(prefetch_futures):
            f.result()
            prefetch_done[0] += 1
            print(f'[{prefetch_done[0]}/{n_repos}] repos prefetched ({prefetch_futures[f]})', flush=True)

    # Phase 2: extract items in parallel (git read ops are safe concurrently)
    all_items = [item for todo in repos_to_process.values() for item in todo]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_and_count, item): item
                   for item in all_items}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                cid = futures[f].get('commit_id', '?')
                logger.debug('extract error %s: %s', cid, e)


def _gen_cpg_batch(entries):
    """Process a batch of (inputDir, outputFile) pairs in a single JVM."""
    fd, worklist_path = tempfile.mkstemp(suffix='.csv', prefix='joern_wl_')
    workdir = tempfile.mkdtemp(prefix='joern_ws_')
    try:
        with os.fdopen(fd, 'w') as f:
            for input_dir, out_file in entries:
                f.write(f'{input_dir},{out_file}\n')
        timeout = max(300, len(entries) * 30)
        r = subprocess.run(
            [path_joern + 'joern'] + _JOERN_JVM_OPTS +
            ['--workspace', workdir,
             '--script', path_locateFunc_batch,
             '--params', f'worklistFile={worklist_path}'],
            cwd=path_joern, timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if r.returncode != 0:
            stderr = r.stderr.decode(errors='ignore')
            if 'OutOfMemoryError' in stderr:
                logger.debug('[OOM] batch of %d entries', len(entries))
            elif entries == entries[:1]:  # log first batch failure for diagnosis
                logger.debug('Batch failed (rc=%d): %s', r.returncode, stderr[:200])
    except subprocess.TimeoutExpired:
        logger.debug('Batch timed out (%d entries)', len(entries))
    finally:
        try:
            os.unlink(worklist_path)
        except OSError:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


def gen_cpg(workers=max(1, _ncpu // 2), batch_size=20):
    dirs = [d for d in os.listdir(path_ab_file) if d != '.DS_Store']

    # filter to items that still need Joern processing
    todo = [d for d in dirs
            if not (os.path.exists(path_ab_file+d+'/cpg_a.txt')
                    and os.path.exists(path_ab_file+d+'/cpg_b.txt'))]

    already = len(dirs) - len(todo)
    if not todo:
        print(f'All {already} CPGs already generated', flush=True)
        return
    print(f'{len(todo)} items to process ({already} already done)', flush=True)

    # build worklist entries and split into batches
    worklist = []
    for d in todo:
        worklist.append((path_ab_file+d+'/a/', path_ab_file+d+'/cpg_a.txt'))
        worklist.append((path_ab_file+d+'/b/', path_ab_file+d+'/cpg_b.txt'))

    # each batch = batch_size items × 2 entries (a + b)
    chunk_sz = batch_size * 2
    chunks = [worklist[i:i+chunk_sz] for i in range(0, len(worklist), chunk_sz)]

    done = 0
    total = len(chunks)
    print(f'Split into {total} batches of ~{batch_size} items, {workers} workers', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_gen_cpg_batch, chunk): chunk for chunk in chunks}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.debug('Batch error: %s', e)
            done += 1
            print(f'[{done}/{total}] batches complete', flush=True)

    # Check how many cpg files were actually created
    created = sum(1 for d in todo
                  if os.path.exists(path_ab_file+d+'/cpg_a.txt')
                  or os.path.exists(path_ab_file+d+'/cpg_b.txt'))
    print(f'gen_cpg: {created}/{len(todo)} items produced cpg files', flush=True)
    if created == 0:
        print('WARNING: No CPG files generated! Joern script may be failing.', flush=True)


def align_cpg(workers=max(1, _ncpu // 2)):
    dirs = [d for d in os.listdir(path_ab_file) if d != '.DS_Store']

    have_cpg = [d for d in dirs
                if (os.path.exists(path_ab_file+d+'/cpg_a.txt')
                    and os.path.exists(path_ab_file+d+'/cpg_b.txt'))]
    align_todo = [d for d in have_cpg
                  if not os.path.exists(path_ab_file+d+'/.aligned')]

    already = len(have_cpg) - len(align_todo)
    if not align_todo:
        print(f'All {already} items already aligned', flush=True)
        return
    print(f'{len(align_todo)} items to align ({already} already done, '
          f'{len(dirs)-len(have_cpg)} without CPGs)', flush=True)

    aligned = 0
    errors = 0
    def _align_one(d):
        locate_and_align(path_ab_file+d+'/')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_align_one, d): d for d in align_todo}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.debug('align error %s: %s', futures[f], e)
                errors += 1
            aligned += 1
    print(f'{aligned}/{len(align_todo)} items aligned ({errors} errors)', flush=True)

def _parse_and_export(src_dir, out_dir, joern_parse, joern_export):
    """Run joern-parse + joern-export for one side (a or b) in an isolated workdir."""
    workdir = tempfile.mkdtemp(prefix='joern_merge_')
    cpg_out = os.path.join(workdir, 'cpg.bin')
    try:
        r1 = subprocess.run(
            [joern_parse] + _JOERN_JVM_OPTS + ['--out', cpg_out, src_dir],
            cwd=path_joern, timeout=_JOERN_TIMEOUT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if r1.returncode != 0:
            stderr = r1.stderr.decode(errors='ignore')
            if 'OutOfMemoryError' in stderr:
                logger.debug('OOM joern-parse: %s', src_dir)
        # Always attempt export — Joern may produce partial output even on non-zero exit
        subprocess.run(
            [joern_export] + _JOERN_JVM_OPTS + ['--repr', 'cpg14', '--out', out_dir, cpg_out],
            cwd=path_joern, timeout=_JOERN_TIMEOUT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.TimeoutExpired:
        logger.debug('joern timed out for %s', src_dir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _merge_cpg_one(cmt):
    if os.path.isfile(cmt.replace('/ab_file/','/data/')+'/out_slim_ninf_noast_n1_w.log'):
        return
    if not (os.path.isfile(cmt+'/cpg_a.txt') and os.path.isfile(cmt+'/cpg_b.txt')):
        return
    joern_parse = path_joern + 'joern-parse'
    joern_export = path_joern + 'joern-export'
    # run a and b in parallel without thread pool overhead
    t = threading.Thread(target=_parse_and_export,
                         args=(cmt+'/a', cmt+'/outA', joern_parse, joern_export))
    t.start()
    _parse_and_export(cmt+'/b', cmt+'/outB', joern_parse, joern_export)
    t.join()
    lenA = os.listdir(cmt+'/outA') if os.path.isdir(cmt+'/outA') else []
    lenB = os.listdir(cmt+'/outB') if os.path.isdir(cmt+'/outB') else []
    if len(lenA)+len(lenB) > 0:
        generateLog(cmt)
    else:
        shutil.rmtree(cmt+'/outA', ignore_errors=True)
        shutil.rmtree(cmt+'/outB', ignore_errors=True)


def merge_cpg(workers=max(1, _ncpu // 2)):
    path_result = path_ab_file.replace('/ab_file/', '/data/')
    os.makedirs(path_result, exist_ok=True)

    all_cmts = [os.path.join(path_ab_file, cmt)
                for cmt in os.listdir(path_ab_file) if cmt != '.DS_Store']

    # filter already-done items upfront
    todo = [cmt for cmt in all_cmts
            if not os.path.isfile(cmt.replace('/ab_file/', '/data/') + '/out_slim_ninf_noast_n1_w.log')]
    already = len(all_cmts) - len(todo)

    if not todo:
        print(f'All {already} CPGs already merged', flush=True)
        return

    # sort largest first to avoid long tail
    todo.sort(key=getdirsize, reverse=True)
    total = len(todo)
    print(f'{total} items to merge ({already} already done)', flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_merge_cpg_one, cmt): cmt for cmt in todo}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.debug('merge error: %s', e)
            done += 1
            if done % 100 == 0 or done == total:
                print(f'[{done}/{total}] CPGs merged', flush=True)

    # Summary
    n_out = sum(1 for d in os.listdir(path_result)
                if os.path.isdir(os.path.join(path_result, d)))
    print(f'Merge complete: {n_out}/{len(all_cmts)} items produced output in {path_result}', flush=True)
    if n_out == 0:
        print('WARNING: No merge output! Joern parse/export may be failing — '
              'try running joern-parse manually to diagnose.', flush=True)


def run_git():
    print('=== Cloning repos ===', flush=True)
    clone_repos()
    print('=== Extracting ab files ===', flush=True)
    get_ab_file()

def run_joern():
    print('=== Generating CPGs ===', flush=True)
    gen_cpg()
    print('=== Aligning CPGs ===', flush=True)
    align_cpg()
    print('=== Merging CPGs ===', flush=True)
    merge_cpg()

if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    log_dir = os.path.join(root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'data_loader_{stage}.log'))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if os.environ.get('VERBOSE') else logging.WARNING)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[console_handler, file_handler],
    )
    if stage == 'clone':
        run_git()
    elif stage == 'cpg':
        run_joern()
    elif stage == 'align':
        align_cpg()
    elif stage == 'merge':
        merge_cpg()
    elif stage == 'all':
        run_git()
        run_joern()
    else:
        print(f'Usage: {sys.argv[0]} [clone|cpg|align|merge|all]', file=sys.stderr)
        sys.exit(1)

