import os
import sys
import time
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, RandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from libs.nets.PGAT import GAT, PGATTrain, UniXCoder

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score, matthews_corrcoef

from libs.preproc import extract_graphs, construct_graphs
from collections import Counter
import torch
import jsonlines
import argparse
import random
import re
from tqdm import tqdm, trange
from torch.utils.data import Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification)

# print(torch.cuda.is_available())
# exit(0)

DATASET = 'spi'

logsPath = './logs/'
mdlsPath = './saved_models/'
if not os.path.exists(logsPath):
    os.mkdir(logsPath)
if not os.path.exists(mdlsPath):
    os.mkdir(mdlsPath)

root = './dataset/'
dataPath = root + 'example_dataset'
# path to the json dataset
jsonlPath = root + 'example_dep.jsonl'
# cache for node info
oldtuplePath = root + 'example_dep_nodeTuple.jsonl'
# cache for sequence
cacheJsonl = root + 'example_{}_nodeTuple_new.jsonl'


logger = logging.getLogger()
logging.basicConfig(filename=logsPath + 'run_example.log',
                    filemode='w',
                    format='%(asctime)s [%(levelname)s]  %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)

# parameters
_CLANG_  = 1

_BATCHSIZE_ = 4
dim_features = 768
num_relations = 20
start_time = time.time() #mark start time

learning_rate = 5e-5
seq_learning_rate = 2e-5
batch_size = _BATCHSIZE_
epoch = 10

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids

def process_line(line):
    pattern = re.compile(r'@@(.*?)\s*@@\s*(.*)')
    new_line = pattern.sub(r'@@\1@@\n\2', line)
    return new_line

def convert_examples_to_features(seq,tokenizer,args):
    """convert examples to token ids"""
    code_lines = seq.splitlines()
    filtered_lines = [line for line in code_lines if not line.startswith('index')]
    processed_lines = [process_line(line) for line in filtered_lines]
    code = '\n'.join(processed_lines)
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# extract graphs
def extractgraphs(path=dataPath):
    cnt = 0
    labelMap = {}
    data = []
    with jsonlines.open(jsonlPath, 'r') as reader:
        for line in reader:
            data.append(line)
    for item in data:
        if not item['commit_id'] == '':
            labelMap[item['commit_id']] = 1 if item['category'] == 'security' else 0
    for root, ds, fs in tqdm(os.walk(path)):
        for file in fs:
            if ('.log' == file[-4:]):
                filename = os.path.join(root, file).replace('\\', '/')
                savename = filename + '_mid.npz'
                cnt += 1
                nodes, edges, nodes0, edges0, nodes1, edges1 = extract_graphs.ReadFile(filename)
                if _CLANG_:
                    nodes = extract_graphs.ProcNodes(nodes, 'PatchCPG')
                    nodes0 = extract_graphs.ProcNodes(nodes0, 'PreCPG')
                    nodes1 = extract_graphs.ProcNodes(nodes1, 'PostCPG')
                label = [labelMap[root[-10:]]]
                np.savez(savename, nodes=nodes, edges=edges, nodes0=nodes0, edges0=edges0, nodes1=nodes1, edges1=edges1,
                         label=label, dtype=object)
    return

# construct graphs
def constructgraphs(path=dataPath):
    for root, ds, fs in tqdm(os.walk(path)):
        for file in fs:
            if ('_mid.npz' == file[-8:]):
                filename = os.path.join(root, file).replace('\\', '/')
                savename = filename.replace('_mid.npz', '_np.npz')
                nodes, edges, nodes0, edges0, nodes1, edges1, label = construct_graphs.ReadFile(filename)
                nodeDict, edgeIndex, edgeAttr = construct_graphs.ProcEdges(edges)
                nodeAttr, nodeInvalid = construct_graphs.ProcNodes(nodes, nodeDict)
                np.savez(savename, edgeIndex=edgeIndex, edgeAttr=edgeAttr, nodeAttr=nodeAttr, label=label,
                         nodeDict=nodeDict)
    return

def transform_first_two(first_two):
    if torch.equal(first_two, torch.tensor([1, 0], dtype=first_two.dtype)):
        return 0
    elif torch.equal(first_two, torch.tensor([0, 1], dtype=first_two.dtype)):
        return 1
    else:
        print(0)
        return 0

def transform_last_three(last_three):
    if torch.equal(last_three, torch.tensor([1, 0, 0], dtype=last_three.dtype)):
        return 0
    elif torch.equal(last_three, torch.tensor([0, 1, 0], dtype=last_three.dtype)):
        return 1
    elif torch.equal(last_three, torch.tensor([0, 0, 1], dtype=last_three.dtype)):
        return 2
    else:
        print(0)
        return 0

def getNodeText(path=None, nodeAttr=None):
    nodeTuple = []
    filename = path.replace('\\', '/')
    nodes, edges, nodes0, edges0, nodes1, edges1 = extract_graphs.ReadFile(filename)
    nodeText  = {}
    for n in nodes:
        nodeText[n[0]]=n[6][0]
    
    nodes, edges, nodes0, edges0, nodes1, edges1, label = construct_graphs.ReadFile(filename+'_mid.npz')
    nodeDict, edgeIndex, edgeAttr = construct_graphs.ProcEdges(edges)
    nodeList = [node[0] for node in nodes]
    nodeOrder = [nodeList.index(node) for node in nodeDict]
    nodesDataNew = [nodes[order] for order in nodeOrder]
    
    for i in range(len(nodesDataNew)):
        nodeData = nodesDataNew[i]
        nodeEmbed = construct_graphs.GetNodeEmbedding(nodeData)
        if nodeAttr[i] in nodeEmbed:
            nodeTuple.append([nodeData[0], nodeAttr[i], nodeText[nodeData[0]]])
        else:
            print('error')
            exit(0)

    return nodeTuple

def batchify_data(data, batch_size):
    # Split the data into batches of size batch_size
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def get_nodeTuple(commit_id, nodeAttr, logfile, nodeTuples, ntKeys):
    if commit_id not in ntKeys:

        nodeTuple = getNodeText(logfile, nodeAttr.tolist()) # [nodeId, nodeAttr, nodeText]
        
        with jsonlines.open(oldtuplePath,'a') as f:
            f.write([commit_id, nodeTuple])
        nodeTuples[commit_id] = nodeTuple
        ntKeys = nodeTuples.keys()
    
    else:
        nodeTuple = nodeTuples[commit_id]
    
    return nodeTuple, nodeTuples, ntKeys

# get dataset
def GetDataset(tokenizer, args, config, path=None, tuplePath=None):
    '''
    Get the dataset from numpy data files.
    :param path: the path used to store numpy dataset.
    :return: dataset - list of torch_geometric.data.Data
    '''
    # check.
    if None == path:
        print('[Error] <GetDataset> Invalid argument \'path\'!')
        return [], []

    details = {}
    jsfile = []
    
    with jsonlines.open(jsonlPath, 'r') as reader:
        for line in reader:
            jsfile.append(line)
    for item in jsfile:
        details[item['commit_id']] = item

    nodeTuples = {}
    if os.path.exists(oldtuplePath):
        with jsonlines.open(oldtuplePath, 'r') as f:
            for item in f:
                nodeTuples[item[0]] = item[1]
    ntKeys = nodeTuples.keys()

    logger.info(f'<GetDataset> Start loading new_nodeTuples {RunTime()}')

    new_nodeTuples = {}
    if os.path.exists(tuplePath):
        with jsonlines.open(tuplePath, 'r') as f:
            for item in f:
                new_nodeTuples[item[0]] = item[1]
    new_ntKeys = new_nodeTuples.keys()
    logger.info(f'<GetDataset> Finish loading new_nodeTuples {RunTime()}')

    # contruct the dataset.
    dataset = []
    files = []

    node_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embeddingmodel = UniXCoder(encoder = node_model, config = config, tokenizer = tokenizer, args =args)
    device = torch.device("cuda:0")
    embeddingmodel.to(device)
    embeddingmodel.eval()

    indexErr_list = ['38970efafd', '5b4dd0f55e', 'eba25057b9', '98d2370413', '97e89ee914', 'b308c82cbd', '14324f585d', 'c834cba905', 'a340046614', 'df1d8a1f29'] # error data on GPU
    
    for root, _, filelist in tqdm(os.walk(path), total=len(os.listdir(path))):

        for file in filelist:
            
            commit_id = root[-10:]
            
            if commit_id in indexErr_list:
                continue

            if file[-7:] == '_np.npz':

                # read a numpy graph file.
                graph = np.load(os.path.join(root, file), allow_pickle=True)
                files.append(os.path.join(root, file[:-7]))
                # sparse each element.
                edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
                nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
                edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)

                if(len(nodeAttr.tolist()) > 3000):
                    logger.debug('delete data ' + commit_id)
                    continue
                    
                new_edgeAttr = edgeAttr

                diff_code = details[commit_id]['diff_code']
                
                logfile = os.path.join(root, file).replace('_np.npz', '')
                label = torch.tensor(graph['label'], dtype=torch.long)

                if commit_id not in new_ntKeys:

                    # 获取[nodeId, nodeAttr, nodeText]
                    try:
                        nodeTuple, nodeTuples, ntKeys = get_nodeTuple(commit_id,nodeAttr,logfile, nodeTuples, ntKeys)
                    except KeyError:
                        logger.debug(f'KeyError: {commit_id}')
                        continue

                    new_data = [item[2] for item in nodeTuple]
                    batches = batchify_data(new_data, 4 * batch_size)

                    all_tokenized_tensors = []
                    for batch in batches:
                        # Tokenize each example in the batch and convert to features
                        tokenized_batch = [convert_examples_to_features(example, tokenizer, args).input_ids for example in batch]
                        
                        # Convert the list of tokenized inputs to a tensor
                        tokenized_tensor = torch.tensor(tokenized_batch)
                        
                        # Ensure the tensor has the correct shape for the model input (batch_size, 512)
                        tokenized_tensor = tokenized_tensor.view(-1, 512).to(device)
                        with torch.no_grad():
                            embedding = embeddingmodel(tokenized_tensor)

                        # Collect the tokenized tensors
                        all_tokenized_tensors.append(embedding)
                    
                    large_tokenized_tensor = torch.cat(all_tokenized_tensors, dim=0)
                    nodeAttr = nodeAttr.to(device)
                    nodeNewAttr = torch.cat([nodeAttr, large_tokenized_tensor], dim=1).tolist()
                    new_nodeTuple = [(*item, nodeNewAttr[index]) for index, item in enumerate(nodeTuple)]
                    
                    # save data
                    with jsonlines.open(tuplePath, 'a') as f:
                        f.write([commit_id, new_nodeTuple])
                    new_nodeTuples[commit_id] = new_nodeTuple
                    new_ntKeys = new_nodeTuples.keys()

                else: 
                    new_nodeTuple = new_nodeTuples[commit_id]
                    
                    nodeNewAttr = [nt[-1] for nt in new_nodeTuple]
                    nodeNewAttr = torch.tensor(nodeNewAttr)
                    nodeNewText = []
                    
                    nodeNewText = [nt[-2] for nt in new_nodeTuple]
                    nodeNewText = '\n\n'.join(nodeNewText)

                nodeNewAttr = torch.tensor(nodeNewAttr)
                nodeNewAttr = nodeNewAttr[:, 20:788]
                input = diff_code
                features = torch.tensor(convert_examples_to_features(diff_code,tokenizer,args).input_ids)

                features = features.view(-1, 512)
                cross_features = features.view(-1, 512)

                data = Data(edge_index=edgeIndex, x=nodeNewAttr, edge_attr=new_edgeAttr, y=label, seq = features, cross_seq = cross_features)
                # append the Data instance to dataset.
                dataset.append(data)

    logger.debug('size of dataset: ' + str(len(dataset)))

    if (0 == len(dataset)):
        print(f'[ERROR] Fail to load data from {path}')
    
    logger.info('finish loading data'+RunTime())
    return dataset, files

def RunTime():
    pTime = ' [TIME: ' + str(round((time.time() - start_time), 2)) + ' sec]'
    return pTime


def evaluate(model, eval_dataset):
    eval_sampler = SequentialSampler(data_source=eval_dataset)
    evalloader = DataLoader(dataset=eval_dataset, batch_size=_BATCHSIZE_, shuffle=False, sampler = eval_sampler)

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    preds = []
    labels = []

    logger.info("***** Running evaluation *****")
    eval_acc, eval_f1, eval_precision, eval_fpr = 0.0, 0.0, 0.0, 1.0

    correct = 0

    for data in evalloader:
        data.to(device)
        out, _, _ = model.forward(data.x, data.edge_index, data.edge_attr, data.batch, data.seq, data.cross_seq)
        pred = out[:,0]>0.5
        correct += int((pred == data.y).sum())
        preds.extend(pred.int().tolist())
        labels.extend(data.y.int().tolist())

    logger.info(f'labels:{labels}')
    logger.info(f'preds:{preds}')

    eval_acc = accuracy_score(labels,preds)
    eval_precision = precision_score(labels,preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)

    conf_matrix = confusion_matrix(labels, preds, labels=[0,1])
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    eval_fpr = FP / ((FP + TN)+0.0001)

    result = {
        'accuracy': eval_acc,
        'precision': eval_precision,
        'recall': eval_recall,
        'f1': eval_f1,
        'fpr': eval_fpr
    }
    
    for key in sorted(result.keys()):
        logger.info("%s = %s", key, str(result[key]))
    logger.info("" + "*" * 30)
    print(f'eval_acc={round(eval_acc,4)}, eval_f1={round(eval_f1, 4)}, eval_precision={round(eval_precision,4)}, eval_fpr={round(eval_fpr, 4)}')

    return result

def test(args, model, config, tokenizer):

    path = './saved_models/model_spi_epoch10_size8_dim768_nr20_acc.pth'

    if not os.path.exists(mdlsPath):
        os.mkdir(mdlsPath)

    model.load_state_dict(torch.load(path))

    testDataset, files = GetDataset(tokenizer, args, config, path=dataPath+'/test', tuplePath = cacheJsonl.format('test'))
    result = evaluate(model=model, eval_dataset=testDataset)
    f1 = result['f1']
    acc = result['accuracy']
    precision = result['precision']
    fpr = result['fpr']
    recall = result['recall']
    print(f'test_acc={round(acc,4)}, test_precision={round(precision,4)}, test_recall={round(recall,4)}, test_f1={round(f1, 4)}, test_fpr={round(fpr, 4)}')
    logger.info(f'test_acc={round(acc,4)}, test_precision={round(precision,4)}, test_recall={round(recall,4)}, test_f1={round(f1, 4)}, test_fpr={round(fpr, 4)}')

def train(args, model, config, tokenizer):
    if not os.path.exists(mdlsPath):
        os.mkdir(mdlsPath)
    
    trainDataset, files = GetDataset(tokenizer, args, config, path=dataPath+'/train', tuplePath = cacheJsonl.format('train'))
    train_data_size = len(trainDataset)
    train_sampler = RandomSampler(data_source=trainDataset)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=_BATCHSIZE_, shuffle=False, sampler = train_sampler)
    
    evalDataset, files = GetDataset(tokenizer, args, config, path=dataPath+'/valid', tuplePath = cacheJsonl.format('valid'))
    
    args.max_steps = args.num_train_epochs * len(trainLoader)

    optimizer_graph = AdamW(
        filter(lambda p: p.requires_grad,
            (list(model.convsA.parameters()) +
                list(model.convsB.parameters()) +
                list(model.convsC.parameters()) +
                list(model.convsD.parameters()) +
                list(model.conv2.parameters()) +
                list(model.conv3.parameters()) +
                list(model.mlp.parameters()) +
                list(model.mlp2.parameters())
                )),
        lr=learning_rate)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_seq = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'encoder' in n and not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if 'encoder' in n and any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if ('multi_classifier' in n or 'cross_att' in n or 'mlp2' in n) and not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if ('multi_classifier' in n or 'cross_att' in n or 'mlp2' in n) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ], lr=seq_learning_rate)

    scheduler_seq = get_linear_schedule_with_warmup(optimizer_seq, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc, best_f1, best_recall, best_precision, best_fpr, best_epoch = 0.0,0.0,0.0,0.0,1.0,-1

    logger.info("***** Running training *****")
    logger.info("Num examples = %d", train_data_size)
    logger.info("Num Epochs = %d", epoch)
    logger.info("Batch size = %d", _BATCHSIZE_)
    logger.info(RunTime())
    last_model = model

    for cur_epoch in range(epoch):
        last_model, loss = PGATTrain(last_model, trainLoader, optimizer_graph, optimizer_seq, scheduler_seq, criterion, cur_epoch, args)
        logger.info(f'epoch {cur_epoch}, train loss = {round(loss, 3)}')

        result = evaluate(model=last_model, eval_dataset=evalDataset)

        if best_acc < result['accuracy']:
            best_acc = result['accuracy']
            best_f1 = result['f1']
            best_precision = result['precision']
            best_recall = result['recall']
            best_fpr = result['fpr']
            best_epoch = cur_epoch
            logger.info(f'best_accuracy change into {best_acc} at epoch {cur_epoch}')
            print(f'best_accuracy change into {best_acc} at epoch {cur_epoch}')
            acc = result['accuracy']
            print(f'best_acc={round(acc,4)}')
            torch.save(last_model.state_dict(), mdlsPath+f'/model_spi_epoch{epoch}_size{train_data_size}_dim{dim_features}_nr{num_relations}_acc.pth')
        logger.info(RunTime())

    logger.info('*********************************************************************************************')
    logger.info(f'Best validation results (epoch {best_epoch}):')
    logger.info(f'  accuracy={round(best_acc,4)}, precision={round(best_precision,4)}, recall={round(best_recall,4)}, f1={round(best_f1,4)}, fpr={round(best_fpr,4)}')
    logger.info('*********************************************************************************************')
    print(f'Best validation results (epoch {best_epoch}): acc={round(best_acc,4)}, precision={round(best_precision,4)}, recall={round(best_recall,4)}, f1={round(best_f1,4)}, fpr={round(best_fpr,4)}')

    logger.info(f'save the last model model_spi_epoch{epoch}_size{train_data_size}_dim{dim_features}_nr{num_relations}_last.pth')
    torch.save(last_model.state_dict(), mdlsPath+f'/model_spi_epoch{epoch}_size{train_data_size}_dim{dim_features}_nr{num_relations}_last.pth')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", default= 'True', action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default= 'True', action='store_true',
                        help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", default= 'False', action='store_true',
                        help="Whether to run test on the test set.")    
    parser.add_argument("--train_batch_size", default=_BATCHSIZE_, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=_BATCHSIZE_, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=epoch, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    # Set seed
    set_seed(args.seed)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels= 1

    seq_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path) 
    model = GAT(num_node_features=dim_features,num_relations=num_relations, encoder = seq_model, config = config, tokenizer = tokenizer, args =args)
    model = model.to(device)

    if args.do_train:
        train(args, model, config, tokenizer)
    if args.do_train or args.do_test:
        test(args, model, config, tokenizer)

if __name__ == '__main__':
    logger.info('Start loading data...'+RunTime())

    extractgraphs()
    constructgraphs()
    
    main()