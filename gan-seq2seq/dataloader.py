import spacy
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import TranslationDataset

from config import DATA_FOLDER, POSITIVE_FILE_EXTENSION, NEGATIVE_FILE_EXTENSION

spacy_en = spacy.load('en_core_web_sm')


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def load_dataset(batch_size, device, path='./data/', pos_ext='.norm', neg_ext='.tox', mode='train'):
    source = Field(tokenize=tokenize_en,
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)
    
    train_data, dev_data, test_data = TranslationDataset.splits(
        path=path,
        exts=(neg_ext, pos_ext),
        fields=(source, source)
    )
    
    source.build_vocab(train_data, min_freq=5)
    
    return source, BucketIterator.splits(
        (train_data, dev_data, test_data),
        shuffle=True if mode=='train' else False,
        batch_size=batch_size,
        device=device)