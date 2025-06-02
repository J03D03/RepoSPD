import os
import subprocess
import sys
import json
import jsonlines
import shutil
import numpy as np
import re

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

if not os.path.exists(path_dots):
    os.mkdir(path_dots)

def get_start_node_id(path):
    nodes, edges, nodes0, edges0, nodes1, edges1= ReadFile(path)
    nodeMax = 0
    for n in nodes:
        nodeMax = max(int(n[0][1:]),nodeMax)
    return nodeMax

def get_dep_log(commit_id, path):
    if not os.path.exists(path_origin_graphs+commit_id+'/out_slim_ninf_noast_n1_w.log'):
        print(f'[INFO] <get_dep_log> {commit_id}: no origin graph')
        return [], [], [], [], [], [], []
    nodeIdStart = get_start_node_id(path_origin_graphs+commit_id+'/out_slim_ninf_noast_n1_w.log')*2

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
        result.append((int(n[0]), int(n[1]), n[2], int(n[3]), n[4], n[6][0]))
    return result

def edge_formatter(edges):
    result = []
    for e in edges:
        result.append((int(e[0]), int(e[1]), e[2], int(e[3])))
    return result

def cpg_formatter(path_origin_graph):
    nodes, edges, nodes0, edges0, nodes1, edges1= ReadFile(path_origin_graph)
    n = node_formatter(nodes)
    n0 = node_formatter(nodes0)
    n1 = node_formatter(nodes1)
    e = edge_formatter(edges)
    e0 = edge_formatter(edges0)
    e1 = edge_formatter(edges1)
    return n, e, n0, e0, n1, e1

def connect_dep(commit_id, path_origin_graph, dep_edges, dep_nodes, AEdges, ANodes, BEdges, BNodes, pre_funcName, post_funcName, rootId)->bool:
    if not os.path.exists(path_origin_graph):
        return False
    nodes, edges, nodes0, edges0, nodes1, edges1= cpg_formatter(path_origin_graph)

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
    if not os.path.exists(path_new_graphs+commit_id):
        os.mkdir(path_new_graphs+commit_id)
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

def main():
    i=0
    data = []
    with jsonlines.open(path_data_dep,'r') as reader:
        for line in reader:
            data.append(line)
    for item in data:
        i+=1
        commit_id = item['commit_id']
        print(f'{i} {commit_id}')

        if os.path.isfile(f'{path_new_graphs}{commit_id}/out_slim_ninf_noast_n1_w_dep.log'):
            print(f'[INFO] <main> CPG {commit_id} exists, continue')
            continue

        if not os.path.exists(f'{path_origin_graphs}{commit_id}'):
            print(f'[INFO] <main> {commit_id} has no origin graph, continue')
            continue

        deps = item['dependency']
        pre_deps = item['pre_dep']
        post_deps = item['post_dep']

        if(len(pre_deps)+len(post_deps)==0):
            print(f'[INFO] <main> {commit_id} has no dependency, continue')
            if os.path.exists(f'{path_origin_graphs}{commit_id}'):
                os.system(f'cp -r {path_origin_graphs}{commit_id} {path_new_graphs}')
            continue

        func_name = deps.keys()
        pre_funcName = pre_deps.keys()
        post_funcName = post_deps.keys()

        outdir = path_dep_c+commit_id
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not os.path.isfile(outdir+'/dependency.c'):
            with open(outdir+'/dependency.c','a') as f:
                for n in func_name:
                    if len(deps[n])>0:
                        f.write(deps[n])
                        f.write('\n')
            if not os.path.getsize(outdir+'/dependency.c'):
                print(f'[INFO] <main> {commit_id}: dependency has no definition, continue')
                shutil.rmtree(outdir)
                if os.path.exists(f'{path_origin_graphs}{commit_id}'):
                    os.system(f'cp -r {path_origin_graphs}{commit_id} {path_new_graphs}')
                continue
        
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


        if not os.path.exists(f'{outdir}/dependency'):
            os.mkdir(f'{outdir}/dependency')

        os.system(f'cd {path_joern}; ./joern-parse {outdir}/pre_dependency.c; ./joern-export --repr cpg14 --out {outdir}/dependency/pre/')

        os.system(f'cd {path_joern}; ./joern-parse {outdir}/post_dependency.c; ./joern-export --repr cpg14 --out {outdir}/dependency/post/')
        
        path_dot = path_dots+commit_id+'/dependency'
        dep_edge, dep_node, AEdges, ANodes, BEdges, BNodes, rootId = get_dep_log(commit_id, path_dot)
        
        if len(dep_edge)+len(dep_node)+len(AEdges)+len(ANodes)+len(BEdges)+len(BNodes)==0:
            print(f'[ERROR] <main> {commit_id}: extract new graph with dependency failed: joern failed')
            if os.path.exists(f'{path_origin_graphs}{commit_id}'):
                os.system(f'cp -r {path_origin_graphs}{commit_id} {path_new_graphs}')
            continue

        path_origin_graph = path_origin_graphs+commit_id+'/out_slim_ninf_noast_n1_w.log'
        flag = connect_dep(commit_id, path_origin_graph, dep_edge, dep_node, AEdges, ANodes, BEdges, BNodes, pre_funcName, post_funcName, rootId)
        if not flag:
            print('[ERROR] <main> no original graph')



if __name__ == '__main__':
    main()