import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from datetime import datetime
import re
tqdm.pandas()

# clean
def rm_ip_address(text):
    return re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)

def rm_link(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def rm_emoji(text):
    emojis = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )
    return emojis.sub(r'', text)

def rm_nonascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)

def rm_inappropriate_sym(text):
    return re.sub(r'[\:\%\=\~\_\n]', ' ', text)

def rm_money(text):
    return re.sub(r'\$\s?((?:\d+[A-z])|((?:\d+[\,\.])+\d+(?=\s))|((?:\d+)))', r'', text)

def space_between_sym(text):    
    return re.sub(r'([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~])', r' \1 ', text)

def rm_additional_space(text):
    return re.sub(r' +', ' ', text)
    
def rm_email(text):
    return re.sub(r'(?:(\S+)?\@\S+)', r'', text)

def rm_middle_dot(text):
    return re.sub(r'(?<=\w)\.(?=\w+)', '', text)

def rm_middle_spaces(text):
    return re.sub(r'(?<=\w)\s(?=\w+)', '', text)

def clean_pipeline(text):
    no_sym = rm_inappropriate_sym(text)
    no_ip_address = rm_ip_address(no_sym)
    no_link = rm_link(no_ip_address)
    no_emoji = rm_emoji(no_link)
    no_nonascii = rm_nonascii(no_emoji)
    no_email = rm_email(no_nonascii)    
    no_mid_dots = rm_middle_dot(no_email)
#     space_between = space_between_sym(no_mid_dots)
    single_space = rm_additional_space(no_mid_dots)

    return single_space

def clean_generated(sentence):
    bos = 'paraphrase: toxic: '
    eos = '</s>'
    # remove 'paraphrase:' suffix
    if bos in sentence:
        sentence = sentence[len(bos):]
    if eos in sentence:
        sentence = sentence[:sentence.index(eos)]
    # split by '>>>' token
    sentence = sentence.split('>>> normal: ')
    # save original sentence
    ori = sentence[0]
    # loop through generation (if any)
    try:
        gen = sentence[1]
        gen = clean_pipeline(gen)
    except:
        gen = ''
        
    return pd.Series([ori, gen], index=['ori', 'gen'])

def choose_from_top_k_top_n(probs, k=50, p=0.8):
    ind = np.argpartition(probs, -k)[-k:]
    top_prob = probs[ind]
    top_prob = {i: top_prob[idx] for idx,i in enumerate(ind)}
    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}
    
    t=0
    f=[]
    pr = []
    for k,v in sorted_top_prob.items():
        t+=v
        f.append(k)
        pr.append(v)
        if t>=p:
            break
    top_prob = pr / np.sum(pr)
    token_id = np.random.choice(f, 1, p = top_prob)

    return int(token_id)


def load_models(model_name, device):
    print ('Loading Trained GPT-2 Model')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.to(device)    
    model.load_state_dict(torch.load(model_name))
    return tokenizer, model


def generate(tokenizer, model, sentences, label):
    with torch.no_grad():
        for idx in range(sentences):
            finished = False
            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to('cpu')
            for i in range(128):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0,-1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy()) #top-k-top-n sampling
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break

            if finished:         
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                print (output_text)
            else:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                print (output_text)


class ParaphraseInferenceDataset(Dataset):
    def __init__(self, path, file_sep, filter_toxic_only=False, source_column='ori', bos='paraphrase: toxic: ', eos='<|endoftext|>'):
        super().__init__()

        self.data_list = []
        self.bos = bos
        self.eos = eos

        data = pd.read_csv(path, sep=file_sep)
        if '.tox' in path:
            data = pd.read_csv(path, sep=file_sep, names=[source_column])

        if filter_toxic_only:
            data = data[data['bt_preds'] == 1]

        for row in tqdm(data.iterrows(), desc=f'Reading {path}'):
            self.data_list.append(f'{self.bos} {row[1][source_column]} >>> normal: ')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


def hf_generate(tokenizer, model, dataset, device, num_beams, do_sample, save_path=None):

    generated = []

    progress_bar = tqdm(range(len(dataset)))

    for item in dataset:
        
        inputs = tokenizer(item, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, num_beams=num_beams, do_sample=do_sample, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated.append(decoded)
        progress_bar.update(1)

    if save_path:
        print(f'saving output to: {save_path}')
        df = pd.DataFrame(generated, columns=['generated'])
        print(f'cleaning data...')
        df_ = df.generated.progress_apply(clean_generated)
        df = pd.concat([df, df_], axis=1)
        df.to_csv(save_path, index=False, header=True, sep='\t')

    return generated


def generate(tokenizer, model, dataset, device, max_length=128, topk=50, topp=0.8, save_path=None):

    generated = []

    with torch.no_grad():
        for item in tqdm(dataset):
            # inputs = tokenizer(item, return_tensors='pt')
            # inputs = {k: v.to(device) for k, v in inputs.items()}
            cur_ids = torch.tensor(tokenizer.encode(item)).unsqueeze(0).to(device)
            for i in range(max_length):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0, -1], dim=0)
                next_token_id = choose_from_top_k_top_n(softmax_logits.cpu().numpy(), k=topk, p=topp)
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break
            
            output_list = list(cur_ids.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated.append(output_text)

    if save_path:
        print(f'saving output to: {save_path}')
        df = pd.DataFrame(generated, columns=['generated'])
        print(f'cleaning data...')
        df_ = df.generated.progress_apply(clean_generated)
        df = pd.concat([df, df_], axis=1)
        df.to_csv(save_path, index=False, header=True, sep='\t')

    return generated

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dn', '--device-number', type=int, action='store')
    parser.add_argument('-mp', '--model-path', type=str, action='store')
    parser.add_argument('--data', type=str, action='store')
    parser.add_argument('-nb', '--num-beam', action='store', type=int)
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('-o', '--output', action='store', type=str)
    parser.add_argument('-sc', '--source-column', type=str, action='store')
    parser.add_argument('--decoding-method', type=str, default='hf')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--topp', type=float, default=0.8)
    parser.add_argument('--maxlength', type=int, default=128)
    args = parser.parse_args()

    DEVICE = f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu'
    NUM_BEAM = args.num_beam
    DO_SAMPLING = args.sampling
    OUTPUT_PATH = args.output

    start_time = datetime.now()
    # load dataset
    dataset = ParaphraseInferenceDataset(path=args.data, file_sep='\t', source_column=args.source_column)
    # load tokenizer and model
    tokenizer, model = load_models(args.model_path, DEVICE)
    # start generation
    print('start generation...')
    if args.decoding_method == 'hf':
        hf_generate(tokenizer, model, dataset, DEVICE, NUM_BEAM, DO_SAMPLING, OUTPUT_PATH)
    elif args.decoding_method == 'self':
        generate(tokenizer, model, dataset, DEVICE, args.maxlength, args.topk, args.topp, OUTPUT_PATH)

    print(f'Finished. Duration: {(datetime.now() - start_time)}')