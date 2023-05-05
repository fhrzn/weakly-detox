from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import evaluate
from sklearn.metrics import accuracy_score
import os
from tqdm.auto import tqdm
import torch
import argparse
import ast
from config import config
from utils import *


def load_dataset(trainpath, devpath, testpath, tokenizer):
    train = pd.read_csv(trainpath, sep='\t')
    dev = pd.read_csv(devpath, sep='\t')
    test = pd.read_csv(testpath, sep='\t')

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train),
        'dev': Dataset.from_pandas(dev),
        'test': Dataset.from_pandas(test)
    })

    try:
        dataset = dataset.remove_columns(['__index_level_0__'])
    except:
        pass

    # process dataset
    dataset_train = dataset['train'].map(tokenize, batched=True, fn_kwargs={'mode': 'train', 'tokenizer': tokenizer})
    dataset_valid = dataset['dev'].map(tokenize, batched=True, fn_kwargs={'mode': 'dev', 'tokenizer': tokenizer})
    dataset_test = dataset['test'].map(tokenize, batched=True, fn_kwargs={'mode': 'test', 'tokenizer': tokenizer})

    # set format
    dataset_train.set_format('torch')
    dataset_valid.set_format('torch')
    dataset_test.set_format('torch')

    return dataset_train, dataset_valid, dataset_test
    

def trainloop(model, optimizer, scheduler, trainloader, devloader, config, num_train_steps, device):
    print('training...')
    tloss = []
    vloss = []
    best_loss = float('inf')

    progressbar = tqdm(range(num_train_steps), desc='Training')

    ############
    # training #
    ############
    for e in range(config['num_epochs']):
        tloss_ = 0
        vloss_ = 0
        model.train()

        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['attention_mask', 'input_ids', 'labels']}
            out = model(**batch)

            tloss_ += out.loss.item()

            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progressbar.update(1)

    ##############
    # validation #
    ##############
    with torch.no_grad():
        for vbatch in tqdm(devloader, desc='Evaluation', leave=True):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['attention_mask', 'input_ids', 'labels']}
            out = model(**batch)

            vloss_ += out.loss.item()

    tloss_ /= len(trainloader)
    vloss_ /= len(devloader)
    tloss.append(tloss_)
    vloss.append(vloss_)

    if vloss_ < best_loss:
        best_loss = vloss_
        torch.save(model.state_dict(), config['model_path'])

    print('Epoch - {}'.format(e + 1))
    print('TLoss: {} | VLoss: {}'.format(tloss_, vloss_))

    # progressbar.close()

    return model

def testloop(model, testloader, config):
    print('testing...')
    testloss = 0

    with torch.no_grad():
        for tbatch in tqdm(testloader, desc='Testing'):
            tbatch = {k: v.to(DEVICE) for k, v in tbatch.items() if k in ['attention_mask', 'input_ids', 'labels']}
            out = model(**tbatch)
            testloss += out.loss.item()
    
    testloss /= len(testloader)
    print('Test Loss: {}'.format(testloss))
    

def main(config, trainpath, devpath, testpath, device):
    print('init model')
    model = AutoModelForSeq2SeqLM.from_pretrained(config['checkpoint'])    
    tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])

    # load data
    print('load dataset & make dataloader')
    train, valid, test = load_dataset(trainpath, devpath, testpath, tokenizer)
    trainloader, validloader, testloader = make_dataloader(train, valid, test, config['batch_size'])

    # init model
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    num_train_steps = config['num_epochs'] * len(trainloader)
    scheduler = get_scheduler('linear',
                              optimizer=optimizer,
                              num_warmup_steps=0,
                              num_training_steps=num_train_steps)
    
    # training!    
    model = trainloop(model, optimizer, scheduler, trainloader, validloader, config, num_train_steps, device)
    # testing
    testloop(model, testloader, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, action='store')
    parser.add_argument('-b', '--batch', type=int, action='store')
    parser.add_argument('-lr', '--learning_rate', type=int, action='store')
    parser.add_argument('-mp', '--model_path', type=str, action='store')
    parser.add_argument('-dn', '--device_number', type=int, action='store')
    parser.add_argument('--data', help='data path', nargs='+', required=True)
    args = parser.parse_args()

    if args.epoch is not None:
        config['num_epochs'] = args.epoch
    
    if args.batch is not None:
        config['batch_size'] = args.batch

    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate

    if args.model_path is not None:
        config['model_path'] = args.model_path

    DEVICE = f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu'

    main(config, args.data[0], args.data[1], args.data[2], DEVICE)
