import re
import ast
from torch.utils.data import DataLoader
import torch

def clean_beginning_symb(text):
    return re.sub(r'^[^A-z0-9\"\']+', '', text)

def tokenize(data, tokenizer, mode='train'):
    source = [clean_beginning_symb(i.strip()) for i in data['source']]
    if mode == 'infer':
        pass
    elif mode == 'train':
        gen = [clean_beginning_symb(i.strip()) for i in data['gen']]
    else:
        gen = []
        first_gen = []
        for sgen in data['gen']:
            try:
                sgen = ast.literal_eval(sgen)
            except:
                print(sgen)
            for i, igen in enumerate(sgen):
                if i == 0:
                    first_gen.append(igen)
                gen.append(clean_beginning_symb(igen.strip()))
                
    if mode == 'infer':
        tokenized = tokenizer(source,
                             max_length=128,
                             truncation=True,
                             padding='max_length')
    else:
        tokenized = tokenizer(source,
                             max_length=128,
                             truncation=True,
                             text_target=gen if mode == 'train' else first_gen,
                             padding='max_length')
    
    return tokenized

def make_dataloader(trainset, validset, testset, batch_size=64):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    return trainloader, validloader, testloader

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)    