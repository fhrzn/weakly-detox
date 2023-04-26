DATASET_NAME = 'Detox-2'
DATA_FOLDER = f'../data/seq2seq'
MODEL_FOLDER = f'./model/{DATASET_NAME}/'

POSITIVE_FILE_EXTENSION = '.norm'
NEGATIVE_FILE_EXTENSION = '.tox'

MODEL_G_AB = 'G_AB'
MODEL_G_BA = 'G_BA'
MODEL_G_AB_PRETRAIN = 'G_AB_pt'
MODEL_G_BA_PRETRAIN = 'G_BA_pt'
MODEL_D_A = 'D_A'
MODEL_D_B = 'D_B'
OPTIMIZER_G_PRETRAIN = 'G_opt'
OPTIMIZER_D_A_PRETRAIN = 'D_A_opt'
OPTIMIZER_D_B_PRETRAIN = 'D_B_opt'

b1 = 0.5  # adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient
enc_emb_dim = 256
dec_emb_dim = 256
d_emb_dim = 256
enc_hid_dim = 512
dec_hid_dim = 512
g_n_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5
g_clip = 1


BATCH_SIZE = 128
START_EPOCH = 0
N_EPOCHS = 50
PRETRAIN_GEN_EPOCH = 10
DECAY_EPOCH = 25
LR = 2e-4
PRETRAIN_LR = 3e-4
L_CYC = 8.0
L_IDT = 5.0
L_ADV = 1.0

LABEL_SMOOTHING = None   # or None
TRAIN_D_ITER = 1
TRAIN_G_ITER = 1
LABEL_SWITCH = False
D_OPTIMIZER = 'SGD' # SGD, RMSProp
STA_CKPT = 's-nlp/roberta_toxicity_classifier'                  # style transfer accuracy
FL_CKPT = 'cointegrated/roberta-large-cola-krishna2020'         # fluency
CP_CKPT = ''        # content preservation



# generate dict config
print('generate config dictionary...')
config = {}
with open('./config.py', 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()

        if 'generate dict config' in line:
            break
        
        if line is None or line == '' or line == '\n':
            continue

        line = line.split(' = ')
        varname = line[0]
        value = line[1]
        config[varname] = eval(value.strip())

# config = {
#     "EXPERIMENT_NAME": 'Detox',
#     "DATA_FOLDER": f'../data/seq2seq',
#     "MODEL_FOLDER": f'./model/{config["EXPERIMENT_NAME"]}/',

#     "POSITIVE_FILE_EXTENSION": '.norm',
#     "NEGATIVE_FILE_EXTENSION": '.tox',

#     "MODEL_G_AB": 'G_AB',
#     "MODEL_G_BA": 'G_BA',
#     "MODEL_G_AB_PRETRAIN": 'G_AB_pt',
#     "MODEL_G_BA_PRETRAIN": 'G_BA_pt',
#     "MODEL_D_A": 'D_A',
#     "MODEL_D_B": 'D_B',
#     "OPTIMIZER_G_PRETRAIN": 'G_opt',
#     "OPTIMIZER_D_A_PRETRAIN": 'D_A_opt',
#     "OPTIMIZER_D_B_PRETRAIN": 'D_B_opt',

#     "b1": 0.5,  # adam: decay of first order momentum of gradient,
#     "b2": 0.999,  # adam: decay of first order momentum of gradient,
#     "enc_emb_dim": 256,
#     "dec_emb_dim": 256,
#     "d_emb_dim": 256,
#     "enc_hid_dim": 512,
#     "dec_hid_dim": 512,
#     "g_n_layers": 2,
#     "enc_dropout": 0.5,
#     "dec_dropout": 0.5,
#     "g_clip": 1,


#     "BATCH_SIZE": 128,
#     "START_EPOCH": 0,
#     "N_EPOCHS": 50,
#     "PRETRAIN_GEN_EPOCH": 10,
#     "DECAY_EPOCH": 25,
#     "PRETRAIN_LR": 3e-4, 
#     "LR": 2e-5,
#     "L_CYC": 8.0,
#     "L_IDT": 5.0,
#     "L_ADV": 1.0,

#     "LABEL_SMOOTHING": 0.9 ,  # or None,
#     "TRAIN_D_ITER": 3,
#     "LABEL_SWITCH": True,
#     "D_OPTIMIZER": 'Adam', # SGD, RMSProp,
#     "STA_CKPT": 's-nlp/roberta_toxicity_classifier',                  # style transfer accuracy,
#     "FL_CKPT": 'cointegrated/roberta-large-cola-krishna2020',         # fluency,
#     "CP_CKPT": ''        # content preservation,
# }