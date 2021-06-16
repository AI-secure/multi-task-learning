#!/usr/bin/env python3

"""
Example for running few-shot algorithms with the PyTorch Lightning wrappers.
"""
import pyximport
pyximport.install() # learn2learn contains some pyx code

import pytorch_lightning as pl
from argparse import ArgumentParser
from learn2learn.algorithms import (
    LightningPrototypicalNetworks,
    LightningMetaOptNet,
    LightningMAML,
    LightningANIL,
)
from pytorch_lightning import loggers as pl_loggers
from lightning_utils import EpisodicBatcher,NoLeaveProgressBar, TrackTestAccuracyCallback
import os
from lightning_mtl import LightningMTL
from models.util import create_model
import torch
from utils import Normalization, ModelCheckpoint, modify_args, print_exp_info
from benchmarks import get_tasksets

HOME = os.path.expanduser('~')
dataset_names = {'mini-imagenet':'miniImageNet',
                 'tiered-imagenet':'tieredImageNet',
                 'cifarfs':'CIFAR-FS',
                 'fc100':'FC100'}
def main():
    parser = ArgumentParser(conflict_handler="resolve", add_help=True)
    # add model and trainer specific args
    parser = LightningPrototypicalNetworks.add_model_specific_args(parser)
    parser = LightningMetaOptNet.add_model_specific_args(parser)
    parser = LightningMAML.add_model_specific_args(parser)
    parser = LightningANIL.add_model_specific_args(parser)
    parser = LightningMTL.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    # add script-specific args
    parser.add_argument("--algorithm", type=str, default="mtl", choices=['anil','maml','metaoptnet','protonet','mtl'])
    parser.add_argument("--dataset", type=str, default="mini-imagenet",choices= ['mini-imagenet', 'tiered-imagenet', 'cifarfs', 'fc100'])
    parser.add_argument("--model",type=str,default='resnet12')
    parser.add_argument("--root", type=str, default="~/data")
    parser.add_argument("--meta_batch_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)

    # pytorch-lignintng has implemented some base flags, which we skip here.
    # e.g., --max_epochs

    parser.add_argument("--log_dir",type=str,default=HOME+"/wandb_logs")
    parser.add_argument("--name",type=str,default="")
    parser.add_argument("--test_method",type=str,default="l2",choices=['default','finetune','l2','tune-all'])
    parser.add_argument('--no_log',default=False,action='store_true')
    parser.add_argument('--drop_rate',default=0.1,type=float, help='Dropout rate. Default in Isola ECCV20')
    parser.add_argument('--optim',default='radam',type=str,choices=['sgd','adam','adamw','adamax','radam','adabound','adamp','sgdw','sgdp'])
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--aug',default='lee',type=str,choices=['norm','lee'], help='data augmentation choice')
    parser.add_argument('--epoch_length',default=1600,type=int)
    parser.add_argument('--val_epoch_length',default=400,type=int)
    parser.add_argument('--test_epoch_length',default=0,type=int,help='if = 0, them no test in training')
    parser.add_argument("--test_epochs", nargs='+', type=int, default=[16,18,21,24,28,34],#[50,55,60,65,70,75,80],
                        help="Epochs to Test. Default is to test every epoch given positive test_epoch_length.")
    parser.add_argument('--final_test_epoch_length',default=2000,type=int)
    parser.add_argument('--save_top_k',default=1,type=int)
    parser.add_argument('--save_last',default=False,action='store_true')
    parser.add_argument('--ckpt_after',type=int,default=0,help='Start checkpointing on and after the selected epoch, if save_top_k = -1, i.e., saving all epochs.')
    parser.add_argument('--resume_train',default=False,action='store_true')
    parser.add_argument('--reset_lr',default=-1,type=float,help='Reset LR when resuming training. val <=0 -> defaulr lr, otherwise take the input value as lr.')
    parser.add_argument('--norm_train_features',default=False,action='store_true', help='Normalize features during training')
    parser.add_argument('--suffix',default='',type=str)
    parser.add_argument('--test_only',default=False,action='store_true')
    parser.add_argument('--load_path',default='',type=str)
    parser.add_argument('--load_epoch',default=-1,type=int)
    parser.add_argument('--stop_patience',default=15,type=int,help='Patience for Early Stopping.')
    parser.add_argument('--test_log_name',default='',type=str)
    parser.add_argument('--test_log_dir',default='logs/finetune/',type=str)
    parser.add_argument('--trainval', default=False,action='store_true',help='Merge the original training and validation set for training, and validate on the test set.')

    args = parser.parse_args()
    modify_args(args) # modify args and the experiment name

    dict_args = vars(args)
    pl.seed_everything(args.seed)
    project_name = f'{dataset_names[args.dataset]}-{args.test_shots}shot'

    print_exp_info(args,project_name)


    # Create tasksets using the benchmark interface
    if args.dataset in ["mini-imagenet", "tiered-imagenet",'cifarfs','fc100']:
        # data_augmentation = "lee2019"
        if args.aug == 'norm':
            data_augmentation = "normalize"
        elif args.aug == 'lee':
            data_augmentation = "lee2019"
            print('Using Lee2019 Data Augmentation, adopted from MetaOptNet (CVPR 2019)')
    elif args.dataset in ['omniglot']:
        data_augmentation = "normalize"
    else:
        raise ValueError()

    # In multi-task learning, we need the original labels of data, instead of the relabelled/reshuffled ones.
    orig_label = args.algorithm == 'mtl'

    tasksets = get_tasksets(
        name=args.dataset,
        train_samples=args.train_queries + args.train_shots,
        train_ways=args.train_ways,
        test_samples=args.test_queries + args.test_shots,
        test_ways=args.test_ways,
        root=args.root,
        data_augmentation=data_augmentation,
        orig_label=orig_label,
        trainval = args.trainval
    )
    episodic_data = EpisodicBatcher(
        tasksets.train,
        tasksets.validation,
        tasksets.test,
        epoch_length= args.epoch_length,
        val_epoch_length = args.val_epoch_length,
        test_epoch_length = args.test_epoch_length,
    )

    model = create_model(name=args.model,n_cls=args.train_ways,dataset=args.dataset,drop_rate=args.drop_rate)
    features = model.features
    if args.norm_train_features:
        features.add_module(str(len(features)), Normalization())
        print('----------------------------')
        print('Normalizing features in training')
    classifier = model.classifier

    # Define loss function
    loss = torch.nn.CrossEntropyLoss(reduction="mean")

    # init algorithm
    if args.algorithm == "protonet":
        algorithm = LightningPrototypicalNetworks(features=features, **dict_args)
    elif args.algorithm == "maml":
        algorithm = LightningMAML(model, **dict_args)
    elif args.algorithm == "anil":
        algorithm = LightningANIL(features, classifier, **dict_args)
    elif args.algorithm == "metaoptnet":
        algorithm = LightningMetaOptNet(features, **dict_args)
    elif args.algorithm == "mtl":
        algorithm = LightningMTL(features, classifier, loss = loss, **dict_args)


    if args.resume_train:
        ckpt_path = os.path.join(args.load_path, 'checkpoints', 'last.ckpt')
        log_id = os.path.basename(args.load_path)
        if not os.path.exists(ckpt_path): assert ValueError(f'No Checkpoint at {ckpt_path}')
    else:
        ckpt_path = None
        log_id = None
    log_name = args.name

    if args.no_log:
        logger = False
    else:
        # We use wandb logger in this project. You could also try other loggers in pytorch_lightning,
        # see more details at https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
        logger = pl_loggers.WandbLogger(
            save_dir=args.log_dir,
            version=None,
            id=log_id,
            name=log_name,
            project=project_name,

        )
        logger.log_hyperparams(dict_args)

    # We use early stopping in this project
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='valid_accuracy',min_delta=0.0,patience=args.stop_patience,verbose=False,mode='max')
    callbacks = [NoLeaveProgressBar(),
                 ModelCheckpoint(monitor='valid_accuracy',
                                              save_top_k=args.save_top_k,
                                              save_last = args.save_last,
                                              mode='max',
                                              ckpt_after=args.ckpt_after
                                              ),
                 early_stopping_callback
                 ]

    if args.test_epoch_length > 0:
        # If given a non-emptry list of test_epoch_length, the trainer will evaluate the model on the test set at the end
        # of these epochs
        callbacks.append(TrackTestAccuracyCallback(test_epochs=args.test_epochs))
        print(f'Will test on epochs {args.test_epochs}, each with {args.test_epoch_length} test tasks')


    # pytorch-lightning trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger = logger,
        gpus=args.gpus,
        max_epochs = args.max_epochs,
        accumulate_grad_batches=args.meta_batch_size,
        callbacks=callbacks,
        resume_from_checkpoint=ckpt_path if args.resume_train else None,
    )


    # -----Training stage------
    algorithm.test_phase = False
    if not args.test_only:
        trainer.fit(model=algorithm, datamodule=episodic_data)


    # -----Testing stage-------
    algorithm.test_phase = True
    # Define the test set
    test_episodic_data = EpisodicBatcher(
        tasksets.test,tasksets.test,tasksets.test,
        epoch_length= args.final_test_epoch_length, # 2000 tasks for test
    )
    # Evaluate on the test set.
    test_result = trainer.test(model=algorithm,ckpt_path="best",datamodule=test_episodic_data)
    test_acc = test_result[0]['test_accuracy']
    print(f'Test Accuracy = {round(test_acc*100,2)}%')

if __name__ == "__main__":
    main()
