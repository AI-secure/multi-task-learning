
import numpy as np
import torch
import os
from torch import nn, optim
import pytorch_lightning as pl
from torch.nn import functional as F
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def turn_off_grad(module:nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def turn_on_grad(module:nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def void_grad(module:nn.Module):
    for p in module.parameters():
        if p.grad is not None:
            p.grad = None

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class Normalization(nn.Module):
    def forward(self, input):
        assert len(input.shape) == 2
        return F.normalize(input)


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies
def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = pred.eq(label).float().mean()
    return accuracy




def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr



class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self,ckpt_after=0,**kwargs):
        super(ModelCheckpoint, self).__init__(**kwargs)
        self.ckpt_after=ckpt_after
    def on_validation_end(self, trainer, pl_module):
        """
        checkpoints can be saved at the end of the val loop
        """
        if (self.save_top_k == -1) and (trainer.current_epoch >= self.ckpt_after):
            self.save_checkpoint(trainer, pl_module)



def modify_args(args):
    if args.dataset == 'omniglot':
        args.train_shots = 1
        args.test_shots = 1


    args.log_dir = f'{args.log_dir}/{args.dataset}-{args.test_shots}shot'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.name == "": args.name = args.algorithm
    args.name += '-' + args.model.replace('resnet','res')

    if args.algorithm == 'mtl':
        if args.train_ways != 5:
            args.suffix += f'{args.train_ways}way-'

    if args.norm_train_features:
        args.name += '-norm'
    args.name += '-' + args.optim
    if args.weight_decay > 0:
        args.name += f'-wd={args.weight_decay}'
    if args.drop_rate != 0.1:
        args.name += f'-drop={args.drop_rate}'
    if len(args.suffix) > 0:
        if args.suffix[-1] == '-':
            args.suffix = args.suffix[:(len(args.suffix)-1)]
        args.name += '-' + args.suffix
    if args.load_path!='':
        args.name += '-'+os.path.basename(args.load_path)

    if args.trainval:
        args.name += '-TrVal'


    if isinstance(args.gpus,int):
        args.gpus = str(args.gpus)

    HOME = os.path.expanduser('~')
    args.log_dir = args.log_dir.replace('~',HOME)
    args.root.replace('~', HOME)

def print_exp_info(args,project_name):
    dict_args = vars(args)
    print('--------------------Args--------------------------')
    for k,v in dict_args.items():
        print(f'{k}: {v}')
    print('--------------------------------------------------')
    print('Project:',project_name)
    print('Run Name',args.name)
    print('Log Path:',args.log_dir)
    print('Algorithm:',args.algorithm)
    print('Dataset:',args.dataset)
    print('GPU:',args.gpus)
