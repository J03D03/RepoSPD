import json
import os
from merge_cpg import merge_cpg,getdirsize, generateLog

root = '/RepoSPD' # Absolute path to `release`
path =            root + '/dataset/example.json' # path of json dataset
path_ab_file =    root + '/data_preproc/ab_file/' # path of ab_file
path_repo =       root + '/data_preproc/repo/' # path of repositories
path_joern =      root + '/joern/'
path_locateFunc = root + '/data_preproc/locateFunc.sc' #script file

# get_ab_file

def get_ab_file():
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if not os.path.exists(path_ab_file):
        os.mkdir(path_ab_file)
    if not os.path.exists(path_repo):
        os.mkdir(path_repo)
    
    for item in data:
        id = item['idx']
        commit_id = item['commit_id']
        print(f'{id} {commit_id}')
        if commit_id == '':
            print('No commit_id, continue')
            continue
        repo = 'FFmpeg' if item['ori_dataset'] in 'ffmpeg' else 'qemu'
        owner = repo # In FFmpeg and qemu, owner is the same as repo name

        content = item['diff_code']

        if not os.path.exists(path_ab_file + commit_id):
            os.system('mkdir ' + path_ab_file + commit_id)

        a = []
        b = []

        # read and parse the diff_code
        i = content.find("diff --git")
        new_content = content[:i]
        content = content[i:]

        # deal with the code difference one by one
        while 'diff --git ' in content:

            i = content.find(' a/')
            j = content.find(' b/')
            k = content.find('\n')

            file_a = content[i + 3:]
            i = file_a.find(' ')
            file_a = file_a[:i]
            file_b = content[j + 3:k]

            if not os.path.exists(path_ab_file + commit_id + '/a'):
                os.system('mkdir ' + path_ab_file + commit_id + '/a')
            if not os.path.exists(path_ab_file + commit_id + '/b'):
                os.system('mkdir ' + path_ab_file + commit_id + '/b')

            # retrive and download the pre- and post-patch files
            if file_a not in a:
                a.append(file_a)
            if file_b not in b:
                b.append(file_b)

            i = content.find('\ndiff --git ')
            if i > 0:
                content = content[i + 1:]
            else:
                # warning: Make sure FFmpeg and qemu are in your path_repo first.
                if not os.path.exists('./repo/'+repo):
                    os.system(f'cd {path_repo}; git clone https://github.com/{owner}/{repo}.git')
                
                os.system('cd '+ path_repo + repo + '; git reset --hard ' + commit_id)
                for f_b in b:
                    os.system('cp '+ path_repo + repo + '/' + f_b + ' ' + path_ab_file + commit_id + '/b/')

                out = os.popen('cd '+ path_repo + repo + '; git rev-list --parents -n 1 ' + commit_id).read()
                commit_a = out[out.find(' ') + 1:].rstrip()
                os.system('cd '+ path_repo + repo + '; git reset --hard ' + commit_a)
                for f_a in a:
                    os.system('cp '+ path_repo + repo + '/' + f_a + ' ' + path_ab_file + commit_id + '/a/')
                break
    
    print('Finish get_ab_file')


# gen_cpg
def gen_cpg():
    dirs = os.listdir(path_ab_file)
    for d in dirs:
        print(d)
        if d == '.DS_Store':	continue

        if os.path.exists(path_ab_file+d+'/cpg_a.txt') and os.path.exists(path_ab_file+d+'/cpg_b.txt'):
            print('cpg exist, skipping')
            continue
        os.system(f'cd {path_joern}; ./joern --script {path_locateFunc} --params inputFile={path_ab_file+d}/a/,outFile={path_ab_file+d}/cpg_a.txt')

        os.system(f'cd {path_joern}; ./joern --script {path_locateFunc} --params inputFile={path_ab_file+d}/b/,outFile={path_ab_file+d}/cpg_b.txt')
        
        os.system('python3 locate_and_align.py '+path_ab_file+d+'/')

def merge_cpg():
    path_result = path_ab_file.replace('/ab_file/', '/data/')

    if not os.path.exists(path_result):
        os.mkdir(path_result)

    commits = os.listdir(path_ab_file)
    files= (os.path.join(path_ab_file, cmt) for cmt in commits if cmt != '.DS_Store')
    commits = sorted(files, key = getdirsize)

    i = 0
    for cmt in commits:
        if os.path.isfile(cmt.replace('/ab_file/','/data/')+'/out_slim_ninf_noast_n1_w.log'):
            print('log file exist, skipping')
            i += 1
            print(i, cmt)
            continue
        if os.path.isfile(cmt+'/cpg_a.txt') and os.path.isfile(cmt+'/cpg_b.txt'):
            os.system(f'cd {path_joern}; ./joern-parse {cmt}/a; ./joern-export --repr cpg14 --out {cmt}/outA')
            os.system(f'cd {path_joern}; ./joern-parse {cmt}/b; ./joern-export --repr cpg14 --out {cmt}/outB')
            lenA = os.listdir(cmt+'/outA')
            lenB = os.listdir(cmt+'/outB')
            if len(lenA)+len(lenB) > 0:
                i += generateLog(cmt)
                print(i, cmt)
            else:
                os.system('rm -r '+cmt+'/outA')
                os.system('rm -r '+cmt+'/outB')
                print('rm', cmt)


if __name__ == '__main__':
    get_ab_file()
    gen_cpg()
    merge_cpg()