import os
import argparse
import torch
import torch.nn as nn
from torch.utils import data as data_utils
#from torch.utils.data.dataset import Dataset
import torch.optim as optim
#from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np
#import utils

from dataset import bAbIDataset
from model1 import REN1

def _gradient_noise_and_clip(parameters,
                                noise_stddev=1e-3, max_clip=40.0):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    nn.utils.clip_grad_norm(parameters, max_clip)

    for p in parameters:
        noise = torch.randn(p.size()) * noise_stddev
        p.grad.data.add_(noise)


def train(model, crit, optimizer, train_loader, args):
    model.train()
    totalloss, correct = 0,0
    sm = torch.nn.Softmax()
    for i, (story, query, answer) in enumerate(train_loader):
        model.zero_grad()
        story, query, answer = story.to(args.device), query.to(args.device), answer.to(args.device)

        preds = model(story, query)
        loss = crit(preds, answer)
        pred_tokens = torch.argmax(sm(preds.detach()), 1)
        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        _gradient_noise_and_clip(model.parameters(),
                noise_stddev=0.005, max_clip=40.0)
        optimizer.step()
        totalloss += loss.item()
        #print(totalloss)
        correct += pred_tokens.eq(answer.detach()).sum().to("cpu").item()


    """
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    norm_type = 2
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)   
    """
    """
    print(total_norm) 
    print(pred_tokens)
    print(answer)
    print(loss)
    print('-------------------')
    """

    totalloss /= (i+1)
    correct = (correct*1.0) / ((i+1) * args.batchsize)
    return {'loss': totalloss,
            'accuracy': correct}

def eval(model, crit, val_loader, args):
    model.eval()
    totalloss, correct = 0,0
    with torch.no_grad():
        for i, (story, query, answer) in enumerate(val_loader):
            story, query, answer = story.to(args.device), query.to(args.device), answer.to(args.device)
            preds = model(story, query)
            loss = crit(preds, answer)
            totalloss += loss.item()
            correct += torch.argmax(preds.detach(), 1).eq(answer.detach()).sum().to("cpu").item()
    
    totalloss /= (i+1)
    correct = (correct*1.0) / ((i+1) * args.batchsize)
    return {'loss': totalloss,
            'accuracy': correct}

def main(args):
    train_dataset = bAbIDataset(args.datadir, args.task)
    val_dataset = bAbIDataset(args.datadir, args.task, train=False)
    print("Dataset size: ", len(train_dataset))
    print("Vocab size: ", train_dataset.num_vocab)
    print(train_dataset.word_idx)
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        num_workers=args.njobs,
        shuffle = False,
        pin_memory=True,
        timeout=300,
        drop_last=True)
    val_loader = data_utils.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        num_workers=args.njobs,
        shuffle = True,
        pin_memory=True,
        timeout=300,
        drop_last=True)
    """
    print(train_dataset[0][0][0])
    print(train_dataset[0][-1][0])
    print([train_dataset.idx2word[i] for i in train_dataset[0][0][0]])
    exit()
    """

    # 2nd and 3rd last arguments for verba and action
    model = REN1(20, train_dataset.num_vocab, 100, args.device, train_dataset.sentence_size)
    #paths =  utils.build_paths(args.output_path, args.exp_name)
    #writer = SummaryWriter(paths['logs'])
    
    # TODO to device
    model = model.to(args.device)
    if args.multi:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_range)

    loss = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)

    start_epoch, end_epoch = 0, args.epochs
    if args.load_model is not None and args.load_model != '':
        pt_model = torch.load(args.load_model)
        try:
            model.load_state_dict(pt_model['state_dict'])
        except:
            model = torch.nn.DataParallel(model, device_ids=[args.gpuid])
            model.load_state_dict(pt_model['state_dict'])
        optimizer.load_state_dict(pt_model['optimizer'])
        start_epoch = pt_model['epochs']
        end_epoch = start_epoch + args.epochs
    

    for epoch in range(start_epoch, end_epoch):
        train_result = train(model, loss, optimizer, train_loader, args)
        val_result = eval(model, loss, val_loader, args)
        if epoch < 200:
            scheduler.step()
    
        #utils.write_logs(epoch, writer, train_result, 'train')
        #utils.write_logs(epoch, writer, val_result, 'val')
        #for param_group in optimizer.param_groups:
        #  writer.add_scalar('lr', param_group['lr'], epoch)
        #  break
        if epoch % args.save_interval == 0 or epoch == args.epochs-1:
            for param_group in optimizer.param_groups:
                log_lr = param_group['lr']
                break
            logline = 'Epoch: [{0}]\t Train Loss {1:.4f} Acc {2:.3f}  \t \
                    Val Loss {3:.4f} Acc {4:.3f} lr {5:.4f}'.format(
                    epoch, train_result['loss'], train_result['accuracy'],
                    val_result['loss'], val_result['accuracy'], log_lr)

            print(logline)
            torch.save({
                'state_dict': model.state_dict(),
                'epochs': epoch+1,
                'args': args,
                'train_scores': train_result,
                'val_scores': val_result,
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.output_path, "%s_%d.pth"%(args.exp_name, epoch)))

    return None

if __name__ == "__main__":
    torch.manual_seed(3000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--datadir", type=str, default='/scratch/ag4508/pn_kaggle/')
    parser.add_argument("--task", type=int, default=1)

    parser.add_argument("--load_model", type=str, default=None,
                              help='Path to saved classifier model')
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--njobs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--output_path", type=str, default='/misc/vlgscratch2/LecunGroup/anant/ren',
                                help='Location to save the logs')
    parser.add_argument("--exp_name", type=str, default='default_model',
                                help='Experiment Name')
    parser.add_argument("--gpuid", type=int, default=0, help='Default GPU id')
    parser.add_argument("--multi", action='store_true', help='To use DataParallel')
    parser.add_argument("--gpu_range",type=str,default="0,1,2,3", help='GPU ids to use if multi')

    args = parser.parse_args()
    args.gpu_range = [int(_) for _ in args.gpu_range.split(",")]
    args.device = torch.device("cuda:%d"%args.gpuid if torch.cuda.is_available() else "cpu")
    #if args.multi:
    #    torch.cuda.set_device(args.gpu_range[0])
    #else:
    #    torch.cuda.set_device(args.gpuid)
    print("Script configuration:\n", args)
    main(args)

