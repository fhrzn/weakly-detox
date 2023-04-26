import torch
import argparse
from gan_helper import get_cycle_gan_network, get_criterions, get_optimizers, get_schedulers
from train_helper import train_generators, train_cycle
from dataloader import load_dataset
from config import *
import sys
import wandb
from utils import get_evaluator

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_generator', action='store_true', default=False)
    parser.add_argument('--load_pretrained_generator', action='store_true', default=False)
    parser.add_argument('--load_pretrained_discriminator', action='store_true', default=False)
    parser.add_argument('-po', '--pretrain_only', action='store_true', default=False)
    parser.add_argument('-dn', '--device_number', action='store', type=int, default=4)
    parser.add_argument('--lr', action='store', type=float)
    parser.add_argument('--pretrain_lr', action='store', type=float)
    parser.add_argument('--lambda_cyc',  action='store')
    parser.add_argument('--lambda_idt',  action='store')
    parser.add_argument('--lambda_adv',  action='store')
    parser.add_argument('--label_smoothing', action='store')
    parser.add_argument('--train_d_iter', action='store')
    parser.add_argument('--label_switch', action='store_true')
    parser.add_argument('--d_optimizer', action='store')
    parser.add_argument('--log', default=None, action='store', type=str)
    parser.add_argument('--wandb_name', default=None, action='store', type=str)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)    
    args = parser.parse_args()
    
    if args.lr:
        config['LR'] = args.lr        

    if args.lambda_cyc:
        config['L_CYC'] = args.lambda_cyc

    if args.lambda_idt:
        config['L_IDT'] = args.lambda_idt

    if args.lambda_adv:
        config['L_ADV'] = args.lambda_adv

    if args.label_smoothing:
        config['LABEL_SMOOTHING'] = args.label_smoothing
    
    if args.label_switch:
        config['LABEL_SWITCH'] = args.label_switch

    if args.d_optimizer:
        config['D_OPTIMIZER'] = args.d_optimizer


    if args.log and args.log == 'wandb':
        wandb.init(config=config, project='cycle-trans-2', name=args.wandb_name)

    # parse args
    DEVICE = f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu'
    LOAD_PRETRAINED_GEN = args.load_pretrained_generator
    LOAD_PRETRAINED_DISC = args.load_pretrained_discriminator
    PRETRAIN_GENERATOR = args.pretrain_generator
    PRETRAIN_GENERATOR_ONLY = args.pretrain_only

    # load dataset
    print('load dataset...')
    source, iterators = load_dataset(BATCH_SIZE, DEVICE)

    G_INPUT_DIM = len(source.vocab)
    G_OUTPUT_DIM = len(source.vocab)

    SOS_IDX = source.vocab.stoi['<sos>']
    PAD_IDX = source.vocab.stoi['<pad>']

    # Losses
    print('get losses...')
    criterion_g_ab, criterion_g_ba, criterion_gan, \
    criterion_discriminator, criterion_cycle, \
    criterion_identity = get_criterions(PAD_IDX, DEVICE)

    # Initialize generator and discriminator
    print('init models...')
    g_ab, g_ba, \
    d_a, d_b = get_cycle_gan_network(G_INPUT_DIM, G_OUTPUT_DIM, DEVICE, PAD_IDX, SOS_IDX,
                                    LOAD_PRETRAINED_GEN, LOAD_PRETRAINED_DISC)

    # watch model gradient
    if wandb.run:
        wandb.watch(g_ab, log_freq=1)
        wandb.watch(g_ba, log_freq=1)
        wandb.watch(d_a, log_freq=1)
        wandb.watch(d_b, log_freq=1)

    # Optimizers
    print('get optimizers...')
    optimizer_g_ab, optimizer_g_ba, \
    optimizer_g, optimizer_d_a, \
    optimizer_d_b = get_optimizers(g_ab, g_ba, d_a, d_b, LR)

    # Learning rate update schedulers
    print('get schedulers...')
    lr_scheduler_g, lr_scheduler_d_a, \
    lr_scheduler_d_b = get_schedulers(optimizer_g, optimizer_d_a, 
                                    optimizer_d_b, START_EPOCH, 
                                    N_EPOCHS, DECAY_EPOCH)

    print('get model evaluator...')
    tokenizer, model = get_evaluator(STA_CKPT, DEVICE)

    if PRETRAIN_GENERATOR:
        print('pretraining generator...')
        train_generators(
            PRETRAIN_GEN_EPOCH,
            g_clip,
            g_ab,
            g_ba,
            iterators,
            criterion_g_ab,
            criterion_g_ba,
            optimizer_g_ab,
            optimizer_g_ba,
            DEVICE
        )

        if PRETRAIN_GENERATOR_ONLY:
            print('system exiting...')
            sys.exit(0)

    print('training cycle gan...')
    train_cycle(
        START_EPOCH,
        N_EPOCHS,
        source,
        DEVICE,
        g_ab,
        g_ba,
        d_a,
        d_b,
        iterators,
        optimizer_g,
        optimizer_d_a,
        optimizer_d_b,
        criterion_identity,
        criterion_gan,
        criterion_cycle,
        criterion_discriminator,
        L_CYC,
        L_IDT,
        L_ADV,
        lr_scheduler_g,
        lr_scheduler_d_a,
        lr_scheduler_d_b,
        tokenizer,
        model
    )


if __name__ == '__main__':
    main()        
