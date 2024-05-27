import argparse
import json
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

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())

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
    parser.add_argument('-d', '--hidden_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('-e', '--epoches', default=10000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('--tasks', default='1p 2p 3p 2i 3i', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")

    return parser.parse_args(args)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, step, average_metrics)

    return all_metrics
        
def load_data(tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    train_queries = []
    test_queries = []
    for task in tasks:
        train_q = pickle.load(open(os.path.join("./data/FB15K-237", f"train-{task}-queries.pkl"), 'rb')).get(task)
        test_q = pickle.load(open(os.path.join("./data/FB15K-237", f"test-{task}-queries.pkl"), 'rb')).get(task)
        
        for q in train_q:
            train_queries.append((q, task))

        for q in test_q:
            train_queries.append((q, task))

    train_answers = pickle.load(open(os.path.join("./data/FB15K-237", "train-answers.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join("./data/FB15K-237", "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join("./data/FB15K-237", "test-easy-answers.pkl"), 'rb'))

    return train_queries, test_queries, train_answers, test_hard_answers, test_easy_answers

def main(args):
    tasks = args.tasks.split('.')

    save_path = os.path.join(os.getcwd(), "models")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open('data/FB15K-237/stats.txt') as f:
        info = f.readlines()
        nentity = int(info[0].split(' ')[1])
        nrelation = int(info[1].split(' ')[1])

    tasks = args.tasks.split(" ")

    train_queries, test_queries, train_answers, test_hard_answers, test_easy_answers = load_data(args, tasks)        

    train_dataloader = DataLoader(
        TrainDataset(
            train_queries, 
            nentity,
            nrelation,
            args.negative_sample_size,
            train_answers
        ),
        batch_size=args.batch_size, 
        collate_fn=TrainDataset.collate_fn
    )

    test_dataloader = DataLoader(
        TestDataset(
            test_queries, 
            args.nentity, 
            args.nrelation, 
        ), 
        batch_size=args.test_batch_size, 
        collate_fn=TestDataset.collate_fn
    )

    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        embedding_range=(args.gamma + args.epsilon) / args.hidden_dim
        query_name_dict = query_name_dict
    )

    if torch.cuda.is_available():
        logging.info("CUDA available, training on GPU ...")
        device = "cuda"
    else:
        logging.warning("CUDA not available, training on CPU ...")
        device = "cpu"

    model = model.to(device)
    
    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate
        )

    if args.do_train:
        training_logs = []
        for step in range(args.epoches):
            log = model.train_step(model, optimizer, args, step, device)

            training_logs.append(log)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step)

    logging.info("Training finished!!")

if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    main(parse_args())