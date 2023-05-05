from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse


def _load_dataset(path):
    if '.txt' in path or '.csv' in path:
        df = pd.read_csv(path, sep='\t', names=['labels', 'text'])
        if df.labels.nunique() <= 2:
            # filter toxic data only
            df = df[df['labels'] == 1]
        else:
            # re-read dataframe
            df = pd.read_csv(path, sep='\t')    # using format columns: [translated, source]
    elif '.tox' in path or '.norm' in path:
        df = pd.read_csv(path, names=['text'], sep='\t')        

    return df
    

def _tokenize_data(batch, tokenizer, max_length=128):
    try:
        text = batch['text']
    except:
        text = batch['translated']          # using format columns: [translated, original]

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )

    return tokenized

def _translate_dataset(model, tokenizer, loader):
    translated = []
    original = []
    source = []

    for batch in tqdm(loader, desc='Translate'):
        try:
            original.extend(batch.pop('text'))
        except:
            # using format columns: [translated, original]
            original.extend(batch.pop('translated'))
            batch.pop('Unnamed: 0')
        if 'original' in batch.keys():
            source.extend(batch.pop('original'))
        # move to device
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        # generate
        batch_generated = model.generate(**batch)
        batch_generated = tokenizer.batch_decode(batch_generated, skip_special_tokens=True)
        translated.extend(batch_generated)        
        
        # empty cache
        torch.cuda.empty_cache()

    if len(source) <= 0:
        return {'translated': translated, 'original': original}
    else:
        # using format columns: [translated, original]
        return {'translated': translated, 'original': original, 'source': source}

def translate_pipeline(trainset, devset, testset, tokenizer, model, batch_size):
    # convert to HF dataset
    try:
        toxics = DatasetDict({
            'train': Dataset.from_pandas(trainset[['text']]),
            'dev': Dataset.from_pandas(devset[['text']]),
            'test': Dataset.from_pandas(testset[['text']])
        })
    except:
        toxics = DatasetDict({
            'train': Dataset.from_pandas(trainset),
            'dev': Dataset.from_pandas(devset),
            'test': Dataset.from_pandas(testset)
        })
    ### attempt to remove unused columns
    try:
        toxics['train'] = toxics['train'].remove_columns(['__index_level_0__'])
        toxics['dev'] = toxics['dev'].remove_columns(['__index_level_0__'])
        toxics['test'] = toxics['test'].remove_columns(['__index_level_0__'])
    except:
        pass    

    # tokenize
    tokenized = toxics.map(_tokenize_data,
                           batched=True,                           
                           fn_kwargs={'tokenizer': tokenizer})
    tokenized.set_format('torch')

    # dataloader
    trainloader = DataLoader(tokenized['train'], shuffle=False, batch_size=batch_size)
    devloader = DataLoader(tokenized['dev'], shuffle=False, batch_size=batch_size)
    testloader = DataLoader(tokenized['test'], shuffle=False, batch_size=batch_size)

    # translate
    traintrans = _translate_dataset(model, tokenizer, trainloader)
    devtrans = _translate_dataset(model, tokenizer, devloader)
    testtrans = _translate_dataset(model, tokenizer, testloader)

    return traintrans, devtrans, testtrans




def main(trainpath, devpath, testpath, outpath, device, batch_size, mode='enru'):
    print(f'got {mode} translation job.')
    # define tokenizer and model
    print('load tokenizers and models...')
    if mode == 'enru':
        ### english to russian
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
        model = model.to(device)

    elif mode == 'ruen':
        ### russian to english
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
        model = model.to(device)
        
    elif mode == 'enid':
        ### english to indonesian
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-id')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-id')
        model = model.to(device)
        
    elif mode == 'iden':
        ### indonesian to english
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-id-en')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-id-en')
        model = model.to(device)
        
    elif mode == 'enfr':
        ### english to french
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        model = model.to(device)
        
    elif mode == 'fren':
        ### french to english
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        model = model.to(device)
        
    elif mode == 'enes':
        ### english to indonesian
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-es')
        model = model.to(device)
        
    elif mode == 'esen':
        ### indonesian to english
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
        model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-es-en')
        model = model.to(device)


    # load data
    print('load data...')
    trainset = _load_dataset(trainpath)
    devset = _load_dataset(devpath)
    testset = _load_dataset(testpath)

    
    # translation pipeline
    print(f'{mode} translation...')
    translate = translate_pipeline(trainset, devset, testset, tokenizer, model, batch_size)

    
    # transform translation to pandas format    
    trans_trainset = pd.DataFrame.from_dict(translate[0])
    trans_devset = pd.DataFrame.from_dict(translate[1])
    trans_testset = pd.DataFrame.from_dict(translate[2])

    print('saving translated data...')
    trans_trainset.to_csv(f'{outpath}/{mode}_train.txt', sep='\t', index=True, header=True)
    trans_devset.to_csv(f'{outpath}/{mode}_valid.txt', sep='\t', index=True, header=True)
    trans_testset.to_csv(f'{outpath}/{mode}_test.txt', sep='\t', index=True, header=True)

    # clean cache
    torch.cuda.empty_cache()
    del model
    del tokenizer

    
    # # ru -> en translation
    # print('ru -> en translation...')
    # ruen_translate = translate_pipeline(ru_trainset, ru_devset, ru_testset, ruen_tokenizer, ruen_model, batch_size)


    # # add to source dataset
    # en_trainset['backtrans'] = ruen_translate[0]
    # en_devset['backtrans'] = ruen_translate[1]
    # en_testset['backtrans'] = ruen_translate[2]


    # # export data
    # print('saving backtranslated data...')
    # en_trainset.to_csv(f'{outpath}/train.txt', sep='\t', index=False, header=True)
    # en_devset.to_csv(f'{outpath}/valid.txt', sep='\t', index=False, header=True)
    # en_testset.to_csv(f'{outpath}/test.txt', sep='\t', index=False, header=True)

    # torch.cuda.empty_cache()
    # del ruen_model
    # del ruen_tokenizer


if __name__ == '__main__':
    # init argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch-size', help='batch size', required=True, type=int)
    parser.add_argument('-dn', '--device-number', help='device number', required=True)
    parser.add_argument('-o', '--output', help='output path', required=True)
    parser.add_argument('-p', '--path', help='data path', nargs='+', required=True)
    parser.add_argument('-m', '--mode', help='mode translation', required=True)
    args = parser.parse_args()

    # define constants
    # TODO: parameterize constants
    DEVICE = f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = args.batch_size

    main(args.path[0], args.path[1], args.path[2], args.output, DEVICE, BATCH_SIZE, args.mode)