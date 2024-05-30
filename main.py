import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from model import KGReasoning
from dataloader import TestDataset, TrainDataset
import time
import pickle
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())

def collate_fn_train(data):
    positive_sample = torch.cat([_[0] for _ in data], dim=0)
    negative_sample = torch.stack([_[1] for _ in data], dim=0)
    subsample_weight = torch.cat([_[2] for _ in data], dim=0)
    query = [_[3] for _ in data]
    query_structure = [_[4] for _ in data]
    return positive_sample, negative_sample, subsample_weight, query, query_structure

def collate_fn_test(data):
    negative_sample = torch.stack([_[0] for _ in data], dim=0)
    query = [_[1] for _ in data]
    query_unflatten = [_[2] for _ in data]
    query_structure = [_[3] for _ in data]
    return negative_sample, query, query_unflatten, query_structure

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=1024, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=24, type=float, help="margin in the distance")
    parser.add_argument('-b', '--batch_size', default=8192, type=int, help="batch size of queries")
    parser.add_argument('-e', '--epoches', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    
    parser.add_argument('--test_batch_size', default=2, type=int, help='valid/test batch size')
    parser.add_argument('--tasks', default='1p 2p 3p 2i 3i', type=str)
    # parser.add_argument('--tasks', default='1p', type=str)

    return parser.parse_args(args)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
def load_data(tasks):
    '''
    Load train, test, valid queries
    '''
    logging.info("Starting loading data")
    valid_queries = pickle.load(open(os.path.join("./FB15k-237-q2b", "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join("./FB15k-237-q2b", "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join("./FB15k-237-q2b", "valid-easy-answers.pkl"), 'rb'))

    train_queries = []
    test_queries = []
    valid_queries = []
    train_answers = defaultdict(set)
    test_hard_answers = defaultdict(set)
    test_easy_answers = defaultdict(set)

    for task in tasks:
        task_name = query_name_dict[task]
        train_q = pickle.load(open(os.path.join("./data/FB15k-237", f"train-{task_name}-queries.pkl"), 'rb')).get(task)
        test_q = pickle.load(open(os.path.join("./data/FB15k-237", f"test-{task_name}-queries.pkl"), 'rb')).get(task)

        assert len(test_q) != 0

        for q in train_q:
            train_queries.append((q, task))

        for q in test_q:
            test_queries.append((q, task))

        new_train_answers = pickle.load(open(os.path.join("./data/FB15k-237", f"train-{task_name}-fn-answers.pkl"), 'rb'))
        new_test_easy_answers = pickle.load(open(os.path.join("./data/FB15k-237", f"test-{task_name}-tp-answers.pkl"), 'rb'))
        new_test_hard_answers = pickle.load(open(os.path.join("./data/FB15k-237", f"test-{task_name}-fn-answers.pkl"), 'rb'))

        train_answers.update(new_train_answers)
        test_easy_answers.update(new_test_easy_answers)
        test_hard_answers.update(new_test_hard_answers)

    return train_queries, train_answers, test_queries, test_hard_answers, test_easy_answers, valid_queries, valid_hard_answers, valid_easy_answers

def construct_dict(queries, query_structures, device):
    batch_queries_dict = defaultdict(list)
    batch_idxs_dict = defaultdict(list)

    for i, query in enumerate(queries):
        batch_queries_dict[query_structures[i]].append(query)
        batch_idxs_dict[query_structures[i]].append(i)
    for query_structure in batch_queries_dict:
        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(device)

    return batch_queries_dict, batch_idxs_dict

def train_step(model, optimizer, positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures, device):
    model.train()
    optimizer.zero_grad()

    batch_size = positive_sample.size(0)

    positive_sample = positive_sample.to(device)
    negative_sample = negative_sample.to(device)
    subsampling_weight = subsampling_weight.to(device)

    batch_queries_dict = defaultdict(list)
    batch_idxs_dict = defaultdict(list)

    batch_queries_dict, batch_idxs_dict = construct_dict(batch_queries, query_structures, device)

    positive_logit, negative_logit, _ = model(positive_sample, negative_sample, batch_queries_dict, batch_idxs_dict)

    negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
    positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
    positive_sample_loss = -(subsampling_weight * positive_score).sum()
    negative_sample_loss = -(subsampling_weight * negative_score).sum()
    positive_sample_loss /= subsampling_weight.sum()
    negative_sample_loss /= subsampling_weight.sum()

    loss = (positive_sample_loss + negative_sample_loss)/ 2

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return loss

def test_step(model, easy_answers, hard_answers, test_dataloader, device, step):
    model.eval()

    metrics_by_structure = defaultdict(list)

    with torch.no_grad():
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=True):

            batch_queries_dict, batch_idxs_dict = construct_dict(queries, query_structures, device)
            negative_sample = negative_sample.to(device)
            
            _, negative_logit, idxs = model(None, negative_sample, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [query_structures[i] for i in idxs]

            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)
            ranking = ranking.scatter_(1, argsort.to(device), torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).to(device)).to(device)

            for idx, (i, query, q_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                hard_answer, easy_answer = hard_answers[query], easy_answers[query]
                num_hard, num_easy = len(hard_answer), len(easy_answer)
                answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)

                cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy

                cur_ranking = cur_ranking - answer_list + 1
                cur_ranking = cur_ranking[masks]

                mrr = torch.mean(1./cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                metrics_by_structure[q_structure].append({
                        'MRR': mrr,
                        'Hit_At1': h1,
                        'Hit_At10': h10
                    })

    for structure in metrics_by_structure:
        num_queries = len(metrics_by_structure[structure])
        mrr = 0
        Hit_At1 = 0
        Hit_At10 = 0
        for metric in metrics_by_structure[structure]:
            mrr += metric["MRR"]
            Hit_At1 += metric["Hit_At1"]
            Hit_At10 += metric["Hit_At10"]
        
        logging.info(f"At {step}, for {structure}, \n \
                    MRR is {mrr / num_queries}, \n \
                    Hit_At1 is {Hit_At1 / num_queries}, \n \
                    Hit_At10 is {Hit_At10 / num_queries}")

def main(args):
    tasks = args.tasks.split('.')

    with open('data/FB15k-237/stats.txt') as f:
        info = f.readlines()
        nentity = int(info[0].split(' ')[1])
        nrelation = int(info[1].split(' ')[1])

    tasks = [name_query_dict[name] for name in args.tasks.split(" ")]

    train_queries, train_answers, test_queries, test_hard_answers, test_easy_answers, valid_queries, valid_hard_answers, valid_easy_answers = load_data(tasks) 

    valid_queries = flatten_query(valid_queries)

    train_dataloader = DataLoader(
        TrainDataset(
            train_queries, 
            nentity,
            nrelation,
            args.negative_sample_size,
            train_answers
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_train
    )

    valid_dataloader = DataLoader(
        TestDataset(
            valid_queries,
            nentity, 
            nrelation, 
        ), 
        batch_size=args.test_batch_size, 
        collate_fn=collate_fn_test
    )

    test_dataloader = DataLoader(
        TestDataset(
            test_queries, 
            nentity, 
            nrelation, 
        ), 
        batch_size=args.test_batch_size, 
        collate_fn=collate_fn_test
    )

    epsilon = 2.0

    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        embedding_range=(args.gamma + epsilon) / args.hidden_dim,
        query_name_dict = query_name_dict,
        gamma = args.gamma
    )

    if torch.cuda.is_available():
        logging.info("CUDA available, training on GPU ...")
        device = "cuda"
    else:
        logging.warning("CUDA not available, training on CPU ...")
        device = "cpu"

    model = model.to(device)
    
    lr = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )

    low_lr_step = 10

    if args.do_train:
        for step in range(args.epoches):
            losses = 0
            iter = 0

            for positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures in tqdm(train_dataloader, disable=True):
                loss = train_step(model, optimizer, positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures, device)
                iter += 1
                losses += loss

            logging.info(f"Losses at epoch {step} is {losses / iter}")
            
            if step % 5 == 0 and step != 0 and args.do_valid:
                test_step(model, valid_easy_answers, valid_hard_answers, valid_dataloader, device, step)
            
            if step == low_lr_step:
                lr = lr / 5
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=lr
                )
                low_lr_step = int(low_lr_step * 2)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_step(model, test_easy_answers, test_hard_answers, test_dataloader, device, -1)

if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    main(parse_args())