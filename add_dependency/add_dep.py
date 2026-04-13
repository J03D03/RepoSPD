import os
os.environ.setdefault('PYTHONUNBUFFERED', '1')
import subprocess
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import jsonlines
import shutil
import tempfile
import numpy as np
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from libs.extract_graphs import ReadFile
from libs.merge_cpg import importCPG, slice

root = '/RepoSPD'
path_ab_file =       root + '/data_preproc/ab_file/'
path_origin_graphs = root + '/data_preproc/data/'
path_dep_c =         root + '/add_dependency/dots/'
path_data_dep =      root + '/dataset/example_dep.jsonl'
path_dots =          root + '/add_dependency/dots/'
path_new_graphs =    root + '/dataset/data/'
path_joern =         root + '/joern/'

os.makedirs(path_dots, exist_ok=True)

def get_start_node_id_from_nodes(nodes):
    nodeMax = 0
    for n in nodes:
        try:
            nodeMax = max(int(n[0][1:]),nodeMax)
        except (ValueError, IndexError):
            continue
    return nodeMax

def get_dep_log(commit_id, path, origin_cpg_data):
    if origin_cpg_data is None:
        print(f'[INFO] <get_dep_log> {commit_id}: no origin graph')
        return [], [], [], [], [], [], []
    raw_nodes = origin_cpg_data[0]
    nodeIdStart = get_start_node_id_from_nodes(raw_nodes)*2

    nodesByFuncA, edgesByFuncA = importCPG(path+'/pre')
    nodesByFuncB, edgesByFuncB = importCPG(path+'/post')
    funcsA = [f for f in nodesByFuncA.keys()]
    funcsB = [f for f in nodesByFuncB.keys()]

    if len(funcsA)+ len(funcsB)==0:
        print(f'[INFO] <get_dep_log> {commit_id}: no dependency')
        return [], [], [], [], [], [], 0

    depEdges, depNodes = [], []
    rootId = {}

    if len(funcsA)>0:
        for f in funcsA:
            newNodes = nodesByFuncA[f]
            newEdges = edgesByFuncA[f]
            rootId[f[1:]] = nodeIdStart+1
            sliceEdges, sliceNodes, nodeID = slice(newNodes, newEdges, [], [], path, -nodeIdStart)
            nodeIdStart = -nodeID
            depEdges.extend(sliceEdges)
            depNodes.extend(sliceNodes)
    if len(funcsB)>0:
        for f in funcsB:
            newNodes = nodesByFuncB[f]
            newEdges = edgesByFuncB[f]
            rootId[f[1:]] = nodeIdStart+1
            sliceEdges, sliceNodes, nodeID = slice([], [], newNodes, newEdges, path, -nodeIdStart)
            nodeIdStart = -nodeID
            depEdges.extend(sliceEdges)
            depNodes.extend(sliceNodes)

    AEdges = []
    ANodes = []
    BEdges = []
    BNodes = []
    for e in depEdges:
        if e[-1] == 0:
            AEdges.append(e)
            BEdges.append(e)
        elif e[-1] == -1:
            AEdges.append(e)
        else:
            BEdges.append(e)
    for n in depNodes:
        if n[1] == 0:
            ANodes.append(n)
            BNodes.append(n)
        elif n[1] == -1:
            ANodes.append(n)
        else:
            BNodes.append(n)

    return depEdges, depNodes, AEdges, ANodes, BEdges, BNodes, rootId

def node_formatter(nodes):
    result = []
    for n in nodes:
        try:
            result.append((int(n[0]), int(n[1]), n[2], int(n[3]), n[4], n[6][0]))
        except (ValueError, IndexError):
            continue
    return result

def edge_formatter(edges):
    result = []
    for e in edges:
        try:
            result.append((int(e[0]), int(e[1]), e[2], int(e[3])))
        except (ValueError, IndexError):
            continue
    return result

def cpg_formatter(path_origin_graph):
    raw_data = ReadFile(path_origin_graph)
    return raw_data

def connect_dep(commit_id, origin_cpg_data, dep_edges, dep_nodes, AEdges, ANodes, BEdges, BNodes, pre_funcName, post_funcName, rootId)->bool:
    if origin_cpg_data is None:
        return False
    raw_nodes, raw_edges, raw_nodes0, raw_edges0, raw_nodes1, raw_edges1 = origin_cpg_data
    nodes = node_formatter(raw_nodes)
    edges = edge_formatter(raw_edges)
    nodes0 = node_formatter(raw_nodes0)
    edges0 = edge_formatter(raw_edges0)
    nodes1 = node_formatter(raw_nodes1)
    edges1 = edge_formatter(raw_edges1)

    funcExist = rootId.keys()
    new_edges0, new_edges1 = [], []
    for n in nodes0:
        for f in pre_funcName:
            if f in n[-1] and f in funcExist:
                new_edges0.append((n[0],-rootId[f],'AST',-1))
    for n in nodes1:
        for f in post_funcName:
            if f in n[-1] and f in funcExist:
                new_edges1.append((n[0],-rootId[f],'AST',1))
    new_edges = new_edges0 + new_edges1

    edges.extend(dep_edges)
    edges.extend(new_edges)
    nodes.extend(dep_nodes)

    edges0.extend(AEdges)
    edges0.extend(new_edges0)
    nodes0.extend(ANodes)

    edges1.extend(BEdges)
    edges1.extend(new_edges1)
    nodes1.extend(BNodes)
    os.makedirs(path_new_graphs+commit_id, exist_ok=True)
    with open(path_new_graphs+commit_id+'/out_slim_ninf_noast_n1_w_dep.log','w') as f:
        f.write('\n'.join(map(str, edges)))
        f.write("\n===========================\n")
        f.write('\n'.join(map(str, nodes)))
        f.write("\n---------------------------\n")
        f.write('\n'.join(map(str, edges0)))
        f.write("\n===========================\n")
        f.write('\n'.join(map(str, nodes0)))
        f.write("\n---------------------------\n")
        f.write('\n'.join(map(str, edges1)))
        f.write("\n===========================\n")
        f.write('\n'.join(map(str, nodes1)))
        f.write("\n")
    print(f'[INFO] <connect_dep> {commit_id}: extract new graph with dependency successfully')
    return True


_JOERN_JVM_OPTS = ['-J-Xmx512m', '-J-XX:TieredStopAtLevel=1', '-J-noverify']

def _joern_parse_export(src_file, out_dir):
    """Run joern-parse + joern-export in an isolated temp workdir."""
    workdir = tempfile.mkdtemp(prefix='joern_dep_')
    joern_parse = path_joern + 'joern-parse'
    joern_export = path_joern + 'joern-export'
    cpg_out = os.path.join(workdir, 'cpg.bin')
    try:
        r1 = subprocess.run(
            [joern_parse] + _JOERN_JVM_OPTS + ['--out', cpg_out, src_file],
            cwd=path_joern, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if r1.returncode != 0:
            stderr = r1.stderr.decode(errors='ignore')
            if 'OutOfMemoryError' in stderr:
                print(f'  [OOM] joern-parse ran out of memory for {src_file} — consider increasing -J-Xmx', flush=True)
        # Always attempt export — Joern may produce partial output even on non-zero exit
        subprocess.run(
            [joern_export] + _JOERN_JVM_OPTS + ['--repr', 'cpg14', '--out', out_dir, cpg_out],
            cwd=path_joern, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.TimeoutExpired:
        print(f'  joern timed out for {src_file}', flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _link_origin(commit_id):
    """Symlink origin graph dir into new_graphs (instant, no copy)."""
    src = f'{path_origin_graphs}{commit_id}'
    dst = f'{path_new_graphs}{commit_id}'
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)


def _process_item(item):
    commit_id = item['commit_id']

    if os.path.isfile(f'{path_new_graphs}{commit_id}/out_slim_ninf_noast_n1_w_dep.log'):
        return

    if not os.path.exists(f'{path_origin_graphs}{commit_id}'):
        return

    deps = item['dependency']
    pre_deps = item['pre_dep']
    post_deps = item['post_dep']

    if(len(pre_deps)+len(post_deps)==0):
        _link_origin(commit_id)
        return

    func_name = deps.keys()
    pre_funcName = pre_deps.keys()
    post_funcName = post_deps.keys()

    outdir = path_dep_c+commit_id
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(outdir+'/dependency.c'):
        with open(outdir+'/dependency.c','a') as f:
            for n in func_name:
                if len(deps[n])>0:
                    f.write(deps[n])
                    f.write('\n')
        if not os.path.isfile(outdir+'/dependency.c') or not os.path.getsize(outdir+'/dependency.c'):
            shutil.rmtree(outdir)
            _link_origin(commit_id)
            return

    if not os.path.isfile(outdir+'/pre_dependency.c'):
        with open(outdir+'/pre_dependency.c','a') as f:
            for n in pre_funcName:
                if len(pre_deps[n])>0:
                    f.write(pre_deps[n])
                    f.write('\n')

    if not os.path.isfile(outdir+'/post_dependency.c'):
        with open(outdir+'/post_dependency.c','a') as f:
            for n in post_funcName:
                if len(post_deps[n])>0:
                    f.write(post_deps[n])
                    f.write('\n')

    os.makedirs(f'{outdir}/dependency', exist_ok=True)

    # Skip Joern if dot output already exists (crash recovery)
    pre_out = outdir+'/dependency/pre/'
    post_out = outdir+'/dependency/post/'
    pre_done = os.path.isdir(pre_out) and os.listdir(pre_out)
    post_done = os.path.isdir(post_out) and os.listdir(post_out)

    # Only run Joern on non-empty source files that haven't been parsed yet
    pre_src = outdir+'/pre_dependency.c'
    post_src = outdir+'/post_dependency.c'
    pre_needed = not pre_done and os.path.isfile(pre_src) and os.path.getsize(pre_src) > 0
    post_needed = not post_done and os.path.isfile(post_src) and os.path.getsize(post_src) > 0

    if pre_needed and post_needed:
        # Run both in parallel without thread pool overhead
        t = threading.Thread(target=_joern_parse_export, args=(pre_src, pre_out))
        t.start()
        _joern_parse_export(post_src, post_out)
        t.join()
    elif pre_needed:
        _joern_parse_export(pre_src, pre_out)
    elif post_needed:
        _joern_parse_export(post_src, post_out)

    # Read origin graph once, reuse for both get_dep_log and connect_dep
    origin_graph_path = path_origin_graphs+commit_id+'/out_slim_ninf_noast_n1_w.log'
    if os.path.exists(origin_graph_path):
        origin_cpg_data = ReadFile(origin_graph_path)
    else:
        origin_cpg_data = None

    path_dot = path_dots+commit_id+'/dependency'
    dep_edge, dep_node, AEdges, ANodes, BEdges, BNodes, rootId = get_dep_log(commit_id, path_dot, origin_cpg_data)

    if len(dep_edge)+len(dep_node)+len(AEdges)+len(ANodes)+len(BEdges)+len(BNodes)==0:
        print(f'[ERROR] <main> {commit_id}: extract new graph with dependency failed: joern failed')
        _link_origin(commit_id)
        return

    flag = connect_dep(commit_id, origin_cpg_data, dep_edge, dep_node, AEdges, ANodes, BEdges, BNodes, pre_funcName, post_funcName, rootId)
    if not flag:
        print(f'[ERROR] <main> {commit_id}: no original graph, copying base graph')
        _link_origin(commit_id)


def main():
    data = []
    with jsonlines.open(path_data_dep,'r') as reader:
        for line in reader:
            data.append(line)

    os.makedirs(path_new_graphs, exist_ok=True)

    # Sanity check: verify origin graphs from stage 2 exist
    if not os.path.isdir(path_origin_graphs):
        print(f'ERROR: Origin graph directory missing: {path_origin_graphs}', flush=True)
        print('Run stage 2 (data_loader.py cpg) first.', flush=True)
        sys.exit(1)
    n_origin = sum(1 for d in os.listdir(path_origin_graphs)
                   if os.path.isdir(os.path.join(path_origin_graphs, d)))
    if n_origin == 0:
        print(f'ERROR: Origin graph directory is empty: {path_origin_graphs}', flush=True)
        print('Run stage 2 (data_loader.py cpg) first.', flush=True)
        sys.exit(1)
    print(f'Found {n_origin} origin graphs in {path_origin_graphs}', flush=True)

    # filter to items that still need processing
    todo = [item for item in data
            if not os.path.isfile(f'{path_new_graphs}{item["commit_id"]}/out_slim_ninf_noast_n1_w_dep.log')]
    already = len(data) - len(todo)
    total = len(todo)
    print(f'{total} items to process ({already} already done)', flush=True)

    if not todo:
        return

    # Pass 1: instant symlinks for no-dep items (no Joern needed)
    need_joern = []
    linked = 0
    for item in todo:
        cid = item['commit_id']
        pre_deps = item.get('pre_dep', {})
        post_deps = item.get('post_dep', {})
        if len(pre_deps) + len(post_deps) == 0:
            _link_origin(cid)
            linked += 1
        else:
            need_joern.append(item)
    print(f'Pass 1: {linked} items symlinked (no deps), {len(need_joern)} items need Joern', flush=True)

    if not need_joern:
        return

    # Pass 2: parallel Joern processing, big items first
    def _dep_size(item):
        try:
            return sum(len(v) for v in item.get('dependency', {}).values()) \
                 + sum(len(v) for v in item.get('pre_dep', {}).values()) \
                 + sum(len(v) for v in item.get('post_dep', {}).values())
        except Exception:
            return 0
    need_joern.sort(key=_dep_size, reverse=True)

    done = 0
    total_joern = len(need_joern)
    _ncpu = int(os.environ.get('CPU_LIMIT', 0)) or os.cpu_count() or 8
    _nworkers = max(1, (_ncpu * 2) // 3)
    print(f'Pass 2: using {_nworkers} workers ({_ncpu} CPUs available)', flush=True)
    with ThreadPoolExecutor(max_workers=_nworkers) as pool:
        futures = {pool.submit(_process_item, item): item for item in need_joern}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                cid = futures[f].get('commit_id', '?')
                print(f'  error {cid}: {e}', flush=True)
            done += 1
            if done % 100 == 0 or done == total_joern:
                print(f'[{done}/{total_joern}] Joern items processed', flush=True)

    # Summary: count actual outputs
    n_out = sum(1 for d in os.listdir(path_new_graphs)
                if os.path.isdir(os.path.join(path_new_graphs, d)))
    print(f'Done: {n_out}/{len(data)} items have RepoCPG output in {path_new_graphs}', flush=True)
    if n_out == 0:
        print('WARNING: No output produced! Check that stage 2 output '
              f'exists in {path_origin_graphs}', flush=True)


if __name__ == '__main__':
    main()
