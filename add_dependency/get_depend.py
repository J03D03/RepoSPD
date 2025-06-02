'''
从diff_code中提取依赖
version 20240628: 
'''
from ast import Pass
import json
import re
import os
import jsonlines
import multiprocessing as mp

from tree_sitter import Language, Parser

root = '/RepoSPD'

json_path =        root + '/dataset/example.json'
repo_path =        root + '/data_preproc/repo/'
cflow_path =       root + '/add_dependency/cflow_files/'
abfile_path =      root + '/data_preproc/ab_file/'
result_path =      root + '/dataset/example_dep.jsonl'
tree_sitter_path = root + '/add_dependency/libs/c-language.so'


def load_cflow(path):
    print(f'Load cflow result from {path}')
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
    print(f'diff_funcs:{diff}')
    return diff, pre_diff, post_diff


def find_func_in_repo(funcs, repo_name, commit_id, preFuncs, postFuncs):
    pre_funcDef = {}
    post_funcDef = {}
    find_funcs = {}
    
    C_LANGUAGE = Language(tree_sitter_path, 'c')
    parser = Parser()
    parser.set_language(C_LANGUAGE)

    os.system(f'cd {repo_path+repo_name}; git reset --hard {commit_id}')
    if len(postFuncs)>0:

        for root, dirs, files in os.walk(repo_path+repo_name):
            for file in files:
                if file.endswith('.c'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path,'r') as f:
                            code = f.read()
                    except FileNotFoundError:
                        print(f'FileNotFoundError in Post-Patch: {file}')
                        continue
                    except UnicodeDecodeError:
                        print(f'UnicodeDecodeError in Post-Patch: {file}')
                        continue
                    try:
                        tree = parser.parse(bytes(code, 'utf-8'))
                        for node in tree.root_node.children:
                            if node.type == 'function_definition':
                                for child in node.children:
                                    if(child.type == 'function_declarator'):
                                        func_name = child.children[0].text.decode('utf-8').strip()
                                        if func_name in postFuncs:
                                            print(file_path)
                                            print('match function definition in post-patch')
                                            print(node.text.decode('utf-8'))
                                            find_funcs[func_name] = node.text.decode('utf-8')
                                            post_funcDef[func_name] = node.text.decode('utf-8')
                    except UnicodeDecodeError:
                        print(f'UnicodeDecodeError in Post-Patch: {file}')
                        continue
    else: print('No postFuncs')
    
    out = os.popen(f'cd {repo_path+repo_name}; git rev-list --parents -n 1 {commit_id}').read()
    pre_patch = out[out.find(' ')+1:].rstrip()
    os.system(f'cd {repo_path+repo_name}; git reset --hard {pre_patch}')
    if len(preFuncs)>0:
        for root, dirs, files in os.walk(repo_path+repo_name):
            for file in files:
                if file.endswith('.c'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path,'r') as f:
                            code = f.read()
                    except FileNotFoundError:
                        print(f'FileNotFoundError in Pre-Patch: {file}')
                        continue
                    except UnicodeDecodeError:
                        print(f'UnicodeDecodeError in Pre-Patch: {file}')
                        continue
                    try:
                        tree = parser.parse(bytes(code, 'utf-8'))
                        for node in tree.root_node.children:
                            if node.type == 'function_definition':
                                for child in node.children:
                                    if(child.type == 'function_declarator'):
                                        func_name = child.children[0].text.decode('utf-8').strip()
                                        if func_name in preFuncs:
                                            print(file_path)
                                            print('match function definition in pre-patch')
                                            print(node.text.decode('utf-8'))
                                            find_funcs[func_name] = node.text.decode('utf-8')
                                            pre_funcDef[func_name] = node.text.decode('utf-8')
                    except UnicodeDecodeError:
                        print(f'UnicodeDecodeError in Pre-Patch: {file}')
                        continue
    else: print('No preFuncs')

    for func in funcs:
        if not func in find_funcs:
            find_funcs[func] = ''
    return find_funcs, pre_funcDef, post_funcDef

def main():
    
    if not os.path.exists(cflow_path):
        os.mkdir(cflow_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    data_finish = []
    if os.path.isfile(result_path):
        with jsonlines.open(result_path,'r') as reader:
            for line in reader:
                data_finish.append(line)
    id_finish = [item['commit_id'] for item in data_finish]
    
    i = 0
    for item in data:
        i+=1
        find_funcs = {}
        pre_findFuncs = {}
        post_findFuncs = {}
        repo = 'FFmpeg' if item['ori_dataset'] in 'ffmpeg' else 'qemu'
        commit_id = item['commit_id']
        if commit_id == '':
            continue
        print(f'{i}: commit_id: {commit_id}   repo: {repo}')
        if commit_id in id_finish:
            continue
        
        path_a = abfile_path + commit_id + '/a/'
        path_b = abfile_path + commit_id + '/b/'
        cf_dir = cflow_path+commit_id
        if not os.path.exists(cf_dir):
            os.mkdir(cf_dir)
        if os.path.exists(path_a):
            a_files = os.listdir(path_a)
            diff_funcs = []
            pre_diffFuncs = []
            post_diffFuncs = []
            for f in a_files:
                if not '.c' == f[-2:]:
                    continue
                afile = path_a + f
                if not os.path.exists(f'{cf_dir}/{f}_a.cflow'):
                    try:
                        os.system(f'cflow -T {afile} >{cf_dir}/{f}_a.cflow')
                    except Exception:
                        pass
                if os.path.exists(path_b):
                    bfile = afile.replace('/a/','/b/')
                    if not os.path.exists(f'{cf_dir}/{f}_b.cflow'):
                        try:
                            os.system(f'cflow -T {bfile} >{cf_dir}/{f}_b.cflow')
                        except Exception:
                            pass 
                    tmp_diff, tmp_pre, tmp_post = load_cflow(f'{cf_dir}/{f}_a.cflow')
                    diff_funcs.extend(tmp_diff)
                    pre_diffFuncs.extend(tmp_pre)
                    post_diffFuncs.extend(tmp_post)
                    diff_funcs = list(set(diff_funcs))
                    pre_diffFuncs = list(set(pre_diffFuncs))
                    post_diffFuncs = list(set(post_diffFuncs))
        else:
            print(f'No ab_file for {commit_id}')

        if os.path.exists(repo_path+repo) and diff_funcs:
            find_funcs, pre_findFuncs, post_findFuncs = find_func_in_repo(diff_funcs,repo,commit_id, pre_diffFuncs, post_diffFuncs)
        
        item['dependency'] = find_funcs
        item['pre_dep'] = pre_findFuncs
        item['post_dep'] = post_findFuncs
        data_finish.append(item)
        with jsonlines.open(result_path,'a') as f:
            f.write(item)

if __name__ == '__main__':
    main()