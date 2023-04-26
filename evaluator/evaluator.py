import pandas as pd
import numpy as np
import torch
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
from pprint import pprint
import argparse
import sys
import evaluate

def load_data(trainpath, devpath, testpath):
    # load dataaset
    if 'bart' not in trainpath and 'cyclegan' not in trainpath:
        train = pd.read_csv(trainpath, sep='\t', index_col=0)
        dev = pd.read_csv(devpath, sep='\t', index_col=0)
        test = pd.read_csv(testpath, sep='\t', index_col=0)
    else:
        train = pd.read_csv(trainpath, sep='\t')
        dev = pd.read_csv(devpath, sep='\t')
        test = pd.read_csv(testpath, sep='\t')


    try:
        train['gen'] = train.gen.fillna('')
        dev['gen'] = dev.gen.fillna('')
        test['gen'] = test.gen.fillna('')
    except:
        pass
    
    # parse filename
    trainname = trainpath.split('/')[-1].split('.')[0]
    devname = devpath.split('/')[-1].split('.')[0]
    testname = testpath.split('/')[-1].split('.')[0]    
    
    # modify column to be more readable
    if 'translated' in train.columns:
        train.rename(columns={'translated': 'gen', 'original': 'translate'}, inplace=True)
        train = train[['source', 'translate', 'gen']]
        dev.rename(columns={'translated': 'gen', 'original': 'translate'}, inplace=True)
        dev = dev[['source', 'translate', 'gen']]
        test.rename(columns={'translated': 'gen', 'original': 'translate'}, inplace=True)
        test = test[['source', 'translate', 'gen']]
    
    elif 'generated' in train.columns:
        train = train[['generated', 'ori', 'gen']]        
        dev = dev[['generated', 'ori', 'gen']]        
        test = test[['generated', 'ori', 'gen']]

    return (train, dev, test), (trainname, devname, testname)

def prepare_data(tokenizer, train, dev, test, BATCH_SIZE):
    # make datasetdict
    toxics = DatasetDict({
        'train': Dataset.from_pandas(train[['gen']]),
        'validation': Dataset.from_pandas(dev[['gen']]),
        'test': Dataset.from_pandas(test[['gen']])
    })
    
    def tokenize_data(batch):
        comment = batch['gen']
                
        tokenized = tokenizer(
            comment,
            truncation=True,
            max_length=128,
            padding='max_length'
        )

        return tokenized
    
    # tokenize
    toxics_tokenized = toxics.map(tokenize_data, batched=True)
    toxics_tokenized.set_format('torch')
    
    # make dataloader
    trainloader = DataLoader(toxics_tokenized['train'], shuffle=False, batch_size=BATCH_SIZE)
    devloader = DataLoader(toxics_tokenized['validation'], shuffle=False, batch_size=BATCH_SIZE)
    testloader = DataLoader(toxics_tokenized['test'], shuffle=False, batch_size=BATCH_SIZE)

    return trainloader, devloader, testloader


def load_and_tokenize(tokenizer, trainpath, devpath, testpath, BATCH_SIZE):  
    # load dataset
    raw_data, filename = load_data(trainpath, devpath, testpath)
    train, dev, test = raw_data

    # prepare data
    loader = prepare_data(tokenizer, train, dev, test, BATCH_SIZE)
    trainloader, devloader, testloader = loader
    
    return (train, dev, test), filename, (trainloader, devloader, testloader)


def predict_toxicity(trainpath, devpath, testpath, outputpath, DEVICE, BATCH_SIZE):
    # load classifier
    MODEL_CKPT = 's-nlp/roberta_toxicity_classifier'   # roberta-base
    id2label = {0: 'Normal', 1: 'Toxic'}
    label2id = {'Normal': 0, 'Toxic': 1}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT,
                                                              num_labels=2,
                                                              id2label=id2label,
                                                              label2id=label2id)    
    model = model.to(DEVICE)
    # model.load_state_dict(torch.load(model_path))
    
    # get dataloader
    raw_data, filenames, loaders = load_and_tokenize(tokenizer, trainpath, devpath, testpath, BATCH_SIZE)
    
    # inference
    trainpreds = []
    devpreds = []
    testpreds = []
    
    # unpack tuples
    trainloader, devloader, testloader = loaders
    train, dev, test = raw_data
    trainname, devname, testname = filenames

    with torch.no_grad():
        for batch in tqdm(trainloader, desc='Trainset Inference'):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            logits = out.logits.squeeze().detach().cpu()
            try:
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, dim=1).numpy()
            except:
                probs = torch.sigmoid(logits).unsqueeze(0)
                preds = torch.argmax(probs, dim=1).numpy()

            trainpreds.extend(preds.tolist())


        for batch in tqdm(devloader, desc='Devset Inference'):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            logits = out.logits.squeeze().detach().cpu()
            try:
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, dim=1).numpy()
            except:
                probs = torch.sigmoid(logits).unsqueeze(0)
                preds = torch.argmax(probs, dim=1).numpy()

            devpreds.extend(preds.tolist())


        for batch in tqdm(testloader, desc='Testset Inference'):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            logits = out.logits.squeeze().detach().cpu()
            try:
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, dim=1).numpy()
            except:
                probs = torch.sigmoid(logits).unsqueeze(0)      # handle single record in a batch
                preds = torch.argmax(probs, dim=1).numpy()

            testpreds.extend(preds.tolist())
        
    train['preds'] = trainpreds
    dev['preds'] = devpreds
    test['preds'] = testpreds
    
    train.to_csv(f'{outputpath}/{trainname}_eval.txt', sep='\t', index=False, header=True)
    dev.to_csv(f'{outputpath}/{devname}_eval.txt', sep='\t', index=False, header=True)
    test.to_csv(f'{outputpath}/{testname}_eval.txt', sep='\t', index=False, header=True)
    
    # clear cache
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
    del model
    del tokenizer
    del loaders
    
    return (train, dev, test), filenames



class SemanticDataset(Dataset):
    def __init__(self, data=None, path=None, base_dir='../data/paraphrase/'):
        
        self.data_list = []
        
        if data is None:
            self.path = os.path.join(base_dir, path)
            data = pd.read_csv(self.path, sep='\t')
            
        for row in tqdm(data.iterrows()):
            try:
                self.data_list.append({
                    'source': row[1].source,
                    'gen': row[1].gen
                })
            except:
                self.data_list.append({
                    'source': row[1].ori,
                    'gen': row[1].gen
                })
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        return self.data_list[item]


def calculate_similarity(raw_data, filenames, outputpath, DEVICE, BATCH_SIZE):
    # load model
    stmodel = SentenceTransformer('all-MiniLM-L6-v2')
    stmodel = stmodel.to(DEVICE)
    
    # unpack tuples
    train, dev, test = raw_data
    trainname, devname, testname = filenames

    # get dataloader
    trainloader = DataLoader(SemanticDataset(train), batch_size=BATCH_SIZE, shuffle=False)
    devloader = DataLoader(SemanticDataset(dev), batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(SemanticDataset(test), batch_size=BATCH_SIZE, shuffle=False)
    
    
    # inference
    train_stsim = []
    dev_stsim = []
    test_stsim = []
    
    for batch in tqdm(trainloader):
        source = batch['source']
        gen = batch['gen']
        src_embed = stmodel.encode(source, convert_to_tensor=True)
        bts_embed = stmodel.encode(gen, convert_to_tensor=True)
        scores = util.cos_sim(src_embed, bts_embed)

        for i in range(len(source)):
            train_stsim.append(scores[i][i].item())

        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
        del src_embed
        del bts_embed    


    for batch in tqdm(devloader):
        source = batch['source']
        gen = batch['gen']
        src_embed = stmodel.encode(source, convert_to_tensor=True)
        bts_embed = stmodel.encode(gen, convert_to_tensor=True)
        scores = util.cos_sim(src_embed, bts_embed)

        for i in range(len(source)):
            dev_stsim.append(scores[i][i].item())

        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
        del src_embed
        del bts_embed
        

    for batch in tqdm(testloader):
        source = batch['source']
        gen = batch['gen']
        src_embed = stmodel.encode(source, convert_to_tensor=True)
        bts_embed = stmodel.encode(gen, convert_to_tensor=True)
        scores = util.cos_sim(src_embed, bts_embed)

        for i in range(len(source)):
            test_stsim.append(scores[i][i].item())

        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
        del src_embed
        del bts_embed
        
    train['sem_similarity'] = train_stsim
    dev['sem_similarity'] = dev_stsim
    test['sem_similarity'] = test_stsim
    
    train.to_csv(f'{outputpath}/{trainname}_eval.txt', sep='\t', index=False, header=True)
    dev.to_csv(f'{outputpath}/{devname}_eval.txt', sep='\t', index=False, header=True)
    test.to_csv(f'{outputpath}/{testname}_eval.txt', sep='\t', index=False, header=True)
    
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
    del stmodel
    
    return (train, dev, test), filenames


def calculate_bleu(raw_data, filenames, outputpath):
    # init evaluate
    bleu = evaluate.load('bleu')

    # unpack tuples
    train, dev, test = raw_data
    trainname, devname, testname = filenames

    # calculate
    train_bleu = []
    dev_bleu = []
    test_bleu = []

    ori_col = 'ori' if 'ori' in train.columns else 'source'
    gen_col = 'gen'    

    for _, row in tqdm(train.iterrows()):
        try:
            bleu_score = bleu.compute(predictions=[row[gen_col].lower()], 
                                    references=[row[ori_col].lower()],
                                    max_order=4)
            train_bleu.append(bleu_score['bleu'])
        except:
            train_bleu.append(0)
    
    for _, row in tqdm(dev.iterrows()):
        try:
            bleu_score = bleu.compute(predictions=[row[gen_col].lower()], 
                                    references=[row[ori_col].lower()],
                                    max_order=4)
            dev_bleu.append(bleu_score['bleu'])
        except:
            dev_bleu.append(0)
    
    for _, row in tqdm(test.iterrows()):
        try:
            bleu_score = bleu.compute(predictions=[row[gen_col].lower()], 
                                    references=[row[ori_col].lower()],
                                    max_order=4)
            test_bleu.append(bleu_score['bleu'])
        except:
            test_bleu.append(0)

    print(len(train), len(train_bleu))
    train['bleu'] = train_bleu
    dev['bleu'] = dev_bleu
    test['bleu'] = test_bleu
    
    train.to_csv(f'{outputpath}/{trainname}_eval.txt', sep='\t', index=False, header=True)
    dev.to_csv(f'{outputpath}/{devname}_eval.txt', sep='\t', index=False, header=True)
    test.to_csv(f'{outputpath}/{testname}_eval.txt', sep='\t', index=False, header=True)
        
    del bleu

    return (train, dev, test), filenames


def calculate_fluency(raw_data, filenames, outputpath, DEVICE, BATCH_SIZE):
    # load model
    tokenizer = AutoTokenizer.from_pretrained('cointegrated/roberta-large-cola-krishna2020')
    model = AutoModelForSequenceClassification.from_pretrained('cointegrated/roberta-large-cola-krishna2020')
    model = model.to(DEVICE)

    # unpack tuples
    train, dev, test = raw_data
    trainname, devname, testname = filenames

    # prepare data
    loader = prepare_data(tokenizer, train, dev, test, BATCH_SIZE)
    trainloader, devloader, testloader = loader

    # inference
    trainfluency = []
    devfluency = []
    testfluency = []

    with torch.no_grad():
        for batch in tqdm(trainloader, desc='Trainset Fluency'):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            logits = out.logits.squeeze().detach().cpu()
            try:
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, dim=1).numpy()
            except:
                probs = torch.sigmoid(logits).unsqueeze(0)
                preds = torch.argmax(probs, dim=1).numpy()

            trainfluency.extend(preds.tolist())

        
        for batch in tqdm(devloader, desc='Devset Fluency'):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            logits = out.logits.squeeze().detach().cpu()
            try:
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, dim=1).numpy()
            except:
                probs = torch.sigmoid(logits).unsqueeze(0)
                preds = torch.argmax(probs, dim=1).numpy()

            devfluency.extend(preds.tolist())

        
        for batch in tqdm(testloader, desc='Testset Fluency'):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            logits = out.logits.squeeze().detach().cpu()
            try:
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, dim=1).numpy()
            except:
                probs = torch.sigmoid(logits).unsqueeze(0)
                preds = torch.argmax(probs, dim=1).numpy()

            testfluency.extend(preds.tolist())

    train['fluency'] = trainfluency
    dev['fluency'] = devfluency
    test['fluency'] = testfluency
    
    train.to_csv(f'{outputpath}/{trainname}_eval.txt', sep='\t', index=False, header=True)
    dev.to_csv(f'{outputpath}/{devname}_eval.txt', sep='\t', index=False, header=True)
    test.to_csv(f'{outputpath}/{testname}_eval.txt', sep='\t', index=False, header=True)
    
    # clear cache
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
    del model
    del tokenizer
    del loader
    
    return (train, dev, test), filenames



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch-size', help='batch size', required=True, type=int)
    parser.add_argument('-dn', '--device-number', help='device number', required=True, type=int)
    parser.add_argument('--data', help='data path', nargs='+', required=True)
    parser.add_argument('-o', '--output', help='output path', required=True)
    args = parser.parse_args()
    
    
    # define constants
    DEVICE = f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = args.batch_size
    
    # predict toxicity  (STA)
    print('predict toxicity...')
    data, filenames = predict_toxicity(args.data[0], args.data[1], args.data[2], args.output, DEVICE, BATCH_SIZE)

    # semantic similarity  (SIM)
    print('calculate semantic similarity...')
    data, filenames = calculate_similarity(data, filenames, args.output, DEVICE, BATCH_SIZE)
    
    # BLEU
    print('calculate bleu...')
    data, filenames = calculate_bleu(data, filenames, args.output)

    # Fluency
    print('calculate fluency...')
    data, filenames = calculate_fluency(data, filenames, args.output, DEVICE, BATCH_SIZE)


    # TODO: Calculate BLEU, Fluency, and Geometric Mean



if __name__ == '__main__':
    main()
