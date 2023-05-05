from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm.auto import tqdm
from config import inference_config as config
from utils import *
import pandas as pd
from datasets import Dataset


def inference(model, loader, device, config, tokenizer):

    generated = []

    for batch in tqdm(loader, desc='Inference'):
        out = model.generate(input_ids=batch['input_ids'].to(device),
                             attention_mask=batch['attention_mask'].to(device),
                             num_beams=config['num_beams'],
                             do_sample=config['do_sample'],
                             min_length=config['min_length'],
                             max_length=config['max_length'])

        generated.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    return generated


def load_dataset(datapath, tokenizer, batch_size):

    data = pd.read_csv(datapath, sep='\t', names=['source'])

    dataset = Dataset.from_pandas(data)

    try:
        dataset = dataset.remove_columns(['__index_level_0__'])
    except:
        pass

    # process dataset
    dataset = dataset.map(tokenize, batched=True, fn_kwargs={'mode': 'infer', 'tokenizer': tokenizer})

    # set format
    dataset.set_format('torch')

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data, dataloader

def main(config, datapath, device):
    print('init model')
    model = AutoModelForSeq2SeqLM.from_pretrained(config['checkpoint'])
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])

    # load model
    print('load model')
    load_model(model, config['model_path'])
    model.eval()

    # load data
    print('load dataset & make dataloader')
    df, loader = load_dataset(datapath, tokenizer, config['batch_size'])

    # inference!
    print('inference...')
    generated = inference(model, loader, device, config, tokenizer)

    # save to df
    df[config['output_column_name']] = generated
    df.to_csv(config['output_path'], sep='\t', index=False, header=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-nb', '--num_beams', type=int, action='store')
    parser.add_argument('-s', '--do_sample', action='store_true', default=True)
    parser.add_argument('-mp', '--model_path', action='store', type=str)
    parser.add_argument('-o', '--output_path', action='store', type=str)
    parser.add_argument('-dn', '--device_number', type=int, action='store')
    parser.add_argument('--data', type=str, action='store')
    args = parser.parse_args()

    if args.num_beams is not None:
        config['num_beams'] = args.num_beams

    if args.do_sample is not None:
        config['do_sample'] = args.do_sample

    if args.model_path is not None:
        config['model_path'] = args.model_path

    if args.output_path is not None:
        config['output_path'] = args.output_path

    DEVICE = f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu'

    main(config, args.data, DEVICE)