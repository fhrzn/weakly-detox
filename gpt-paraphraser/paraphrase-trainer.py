import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import os
from tqdm.auto import tqdm
import torch
import argparse


class UnparallelDataset(Dataset):
    def __init__(self, path, data_dir='../data/classification/skoltech-jigsaw/'):
        super().__init__()
        
        data_path = os.path.join(data_dir, path)
        
        self.data_list = []
        self.eos = " </s>"
        
        df = pd.read_csv(data_path, sep='\t', names=['labels', 'text'])
        df['labels'] = df.labels.apply(lambda x: 'toxic' if x == 1 else 'normal')
        
        for row in tqdm(df.iterrows(), desc=f'Reading {path}'):
            self.data_list.append(f'{row[1]["labels"]}: {row[1]["text"]}{self.eos}')
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        return self.data_list[item]
    

class ParaphraseDataset(Dataset):
    def __init__(self, path, data_dir='../data/paraphrase/'):
        super().__init__()
        
        data_path = os.path.join(data_dir, path)
        
        self.data_list = []
        self.eos = " </s>"
        
        df = pd.read_csv(data_path, sep='\t')
        
        for row in tqdm(df.iterrows(), desc=f'Reading {path}'):
            self.data_list.append(f'paraphrase: toxic: {row[1]["source"]} >>> normal: {row[1]["backtranslate"]}{self.eos}')
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        return self.data_list[item]


def get_data_loader(path, mode='unparallel'):
    print('get dataloader...')
    if mode == 'unparallel':
        try:
            dataset = UnparallelDataset(path)
        except:
            dataset = UnparallelDataset(path, data_dir='../data/classification/skoltech-jigsaw/')
    elif mode == 'paraphrase':
        try:
            dataset = ParaphraseDataset(path)
        except:
            dataset = ParaphraseDataset(path, data_dir='../data/paraphrase/')

    else:
        raise NotImplementedError('available mode: [unparallel, paraphrase]')
        
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader


def train(epochs, loader, batch_size, tokenizer, model, device, MODEL_NAME):
    
    batch_counter = 0
    sumloss = 0
    bestloss = float('inf')

    num_steps = epochs * len(loader)
    pb = tqdm(range(num_steps))
    
    for e in range(epochs):
        print(f'Epoch {e+1}')
        
        for step, txt in enumerate(loader):
            txt = torch.tensor(tokenizer.encode(txt[0]))
            txt = txt.unsqueeze(0).to(device)
            outputs = model(txt, labels=txt)
            loss, _ = outputs[:2]
            loss.backward()
            sumloss += loss.item()
            
            if step % batch_size == 0:
                batch_counter += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
                
            if batch_counter == 10:
                print(f'Total Loss: {sumloss}')
                if sumloss <= bestloss:
                    bestloss = sumloss
                    save_model(model, MODEL_NAME)
                batch_counter = 0
                sumloss = 0
                
            pb.update(1)
                
    return model


def save_model(model, name):
    print('saving model...')
    torch.save(model.state_dict(), f'{name}')
    
    
def load_models():
    print('load models...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    return tokenizer, model


if __name__ == '__main__':
    # init argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-e', '--epoch', default= 3,type=int, action='store', help='Number of epochs to run')
    parser.add_argument('-w', '--warmup', default=300, type=int, action='store', help='Number of warmup steps to run')
    parser.add_argument('-mp', '--model-path', default='mymodel.pt', type=str, action='store', help='Name of the model file')
    parser.add_argument('--data', type=str, action='store', help='Name of the data file')
    parser.add_argument('-m', '--mode', type=str, action='store')
    parser.add_argument('-b', '--batch', type=int, default=32, action='store', help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=3e-5, type=float, action='store', help='Learning rate for the model')
    parser.add_argument('--seqlen', default=128, type=int, action='store', help='Maximum length of sequence')
    parser.add_argument('-dn', '--device-number', type=int, action='store')
    args = parser.parse_args()

    EPOCHS = args.epoch
    WARMUP_STEPS = args.warmup
    MODEL_NAME = args.model_path
    DATA_FILE = args.data
    MODE = args.mode
    BATCH_SIZE = args.batch
    LEARNING_RATE = args.learning_rate
    MAX_SEQ_LEN = args.seqlen
    DEVICE_NUMBER = args.device_number

    TOKENIZER, MODEL = load_models()
    LOADER = get_data_loader(DATA_FILE, MODE)

    DEVICE = f'cuda:{DEVICE_NUMBER}' if torch.cuda.is_available() else 'cpu'

    num_training_steps = EPOCHS * len(LOADER)
    model = MODEL.to(DEVICE)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps)

    model = train(EPOCHS, LOADER, BATCH_SIZE, TOKENIZER, MODEL, DEVICE, MODEL_NAME)
    # save_model(model, MODEL_NAME)
