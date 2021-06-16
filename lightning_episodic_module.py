#!/usr/bin/env python3

"""
"""

import pytorch_lightning as pl

from torch import optim
from argparse import ArgumentParser
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import torch_optimizer
from sklearn.neighbors import NearestCentroid
from learn2learn.utils import accuracy
from scipy.special import softmax
from copy import deepcopy
def tensors2arrays(*tensors):
    '''Convert pytorch tensors to numpy arrays'''
    out = []
    for tensor in tensors:
        out.append(tensor.detach().cpu().numpy())
    return tuple(out)

def get_query_support(data,labels, ways,shots,queries,detach_features:bool,query_support_split:bool,
                      to_numpy=False,flatten_labels=False):
    '''Randomly split a set into query and support data following the given ways, shots, (number of) queries'''
    if query_support_split:
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shots + queries)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        support = data[support_indices]
        if detach_features:
            support = support.detach()

        support_labels = labels[support_indices]
        query = data[query_indices]
        query_labels = labels[query_indices]
    else:
        if detach_features:
            support = data.detach()
        query = data
        query_labels,support_labels = labels,labels
    if flatten_labels:
        query_labels = query_labels.view(-1)
        support_labels = support_labels.view(-1)

    if to_numpy:
        query, query_labels, support, support_labels = tensors2arrays(query, query_labels, support, support_labels)
    return query, query_labels, support, support_labels

def normalize(x):
    '''Normalize each feature vector to have a unit norm'''
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def adjust_learning_rate(epoch, decay_epochs, decay_rate, lr, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(decay_epochs))
    if steps > 0:
        new_lr = lr * (decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

class LightningEpisodicModule(pl.LightningModule):
    """docstring for LightningEpisodicModule"""

    train_shots = 5
    train_queries = 5
    train_ways = 5
    test_shots = 5
    test_queries = 30 # should be <= 20 - shots for Omniglot
    test_ways = 5
    test_l2 = 3 # The inverse of L2 regularization on the linear classifier in the fine-tuning stage for validation/test
    test_steps = 10
    lr = 0.001
    lr_decay_patience = 4
    momentum = 0.9
    scheduler = 'plateau' # Use the ReduceOnPlateau learning rate scheduler
    scheduler_step = 20
    scheduler_decay = 1.0
    weight_decay = 0
    decay_rate = 0.1
    decay_epochs = [75,110]

    def __init__(self,**kwargs):
        super().__init__()
        assert len(kwargs)>0
        self.weight_decay = kwargs.get("weight_decay",LightningEpisodicModule.weight_decay)
        self.norm_test_features = kwargs.get("norm_test_features",True)
        # self.norm_train_features = kwargs.get("norm_train_features")
        self.scheduler = kwargs.get('scheduler',LightningEpisodicModule.scheduler)
        self.decay_rate = kwargs.get('decay_rate',LightningEpisodicModule.decay_rate)
        self.decay_epochs= kwargs.get('decay_epochs',LightningEpisodicModule.decay_epochs)
        self.momentum = LightningEpisodicModule.momentum
        self.optim = kwargs['optim']
        self.test_method = kwargs.get('test_method','l2')
        self.test_l2 = kwargs.get('test_l2',LightningEpisodicModule.test_l2)
        self.lr_decay_patience = kwargs.get('decay_patience',LightningEpisodicModule.lr_decay_patience)
        self.test_lr = kwargs.get('test_lr',LightningEpisodicModule.lr)
        self.test_steps = kwargs.get('test_steps', LightningEpisodicModule.test_steps)
        self.test_weight_decay = kwargs.get('test_weight_decay',0)
        if self.test_method != 'default':
            print('=====================================================')
            print(f'Test Method: {self.test_method.capitalize()}')
        if self.lr_decay_patience != LightningEpisodicModule.lr_decay_patience:
            print('--------------------------------------------')
            print(f'LR Decaying Patience = {self.lr_decay_patience}')
        self.hyperparams = {'weight_decay':self.weight_decay,
                            'norm_test_features':self.norm_test_features,
                            # 'norm_train_features':self.norm_train_features,
                            'scheduler':self.scheduler,
                            'decay_rate':self.decay_rate,
                            'momentum':self.momentum,
                            'decay_epochs':self.decay_epochs,
                            'test_method':self.test_method,

                            }

        if self.norm_test_features:
            print('Will normalize features in validation/test')
            print('-------------------------------------------')
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler="resolve",
        )
        parser.add_argument(
            "--train_ways", type=int, default=LightningEpisodicModule.train_ways
        )
        parser.add_argument(
            "--train_shots", type=int, default=LightningEpisodicModule.train_shots
        )
        parser.add_argument(
            "--train_queries", type=int, default=LightningEpisodicModule.train_queries
        )
        parser.add_argument(
            "--test_ways", type=int, default=LightningEpisodicModule.test_ways
        )
        parser.add_argument(
            "--test_shots", type=int, default=LightningEpisodicModule.test_shots
        )
        parser.add_argument(
            "--test_queries", type=int, default=LightningEpisodicModule.test_queries
        )
        parser.add_argument("--lr", type=float, default=LightningEpisodicModule.lr)
        parser.add_argument(
            "--scheduler_step", type=int, default=LightningEpisodicModule.scheduler_step
        )
        parser.add_argument(
            "--scheduler_decay",
            type=float,
            default=LightningEpisodicModule.scheduler_decay,
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=LightningEpisodicModule.weight_decay
        )
        parser.add_argument(
            "--norm_test_features",
            default=True,
            action='store_true',
            help='By default, we normalize features during test (for L2 test method).'
        )

        parser.add_argument(
            '--decay_rate',
            default=LightningEpisodicModule.decay_rate,
            type=float
        )
        parser.add_argument(
            '--decay_epochs',
            default=LightningEpisodicModule.decay_epochs,
            type=int,
            nargs='+'
        )
        parser.add_argument(
            '--scheduler',
            type=str,
            default=LightningEpisodicModule.scheduler
        )
        parser.add_argument(
            '--test_l2',
            type=float,
            default=LightningEpisodicModule.test_l2
        )
        parser.add_argument(
            '--test_lr',
            type=float,
            default=LightningEpisodicModule.lr
        )
        parser.add_argument(
            '--test_steps',
            type=int,
            default=LightningEpisodicModule.test_steps
        )
        parser.add_argument(
            '--decay_patience',
            type=int,
            default=LightningEpisodicModule.lr_decay_patience,
            help='Patience for Learning Rate Decay'
        )
        parser.add_argument(
            '--test_weight_decay',
            type=float,
            default=0
        )
        # parser.add_argument('--test-method',type=str, default='l2',choices=['l2','knn'])
        return parser

    def training_step(self, batch, batch_idx):
        train_loss, train_accuracy = self.meta_learn(
            batch, batch_idx, self.train_ways, self.train_shots, self.train_queries
        )
        self.log(
            "train_loss",
            train_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_accuracy",
            train_accuracy.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        if self.test_method == 'default':
            valid_loss, valid_accuracy = self.meta_learn(
                batch, batch_idx, self.test_ways, self.test_shots, self.test_queries
            )
        else:
            # For multi-task learning, we use test_method = 'l2'
            valid_loss, valid_accuracy = self.meta_test(
                batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,method=self.test_method
            )
        self.log(
            "valid_loss",
            valid_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "valid_accuracy",
            valid_accuracy.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return valid_loss.item()


    def test_step(self, batch, batch_idx):

        if self.test_method == 'default':
            # The default test method is just use the original meta-learning function (e.g., it's SGD adaptation for MAML)
            # For multi-task learning, we do not use this meta_learn function for test.
            test_loss, test_accuracy = self.meta_learn(
                batch, batch_idx, self.test_ways, self.test_shots, self.test_queries
            )
        else:
            # For non-default test methods, we invoke the meta_test function and pass the test_method as an argument
            test_loss, test_accuracy = self.meta_test(batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,
                                                      method=self.test_method)
        self.log(
            "test_loss",
            test_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_accuracy",
            test_accuracy.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return test_loss.item()

    # @torch.no_grad()
    @torch.enable_grad()
    def meta_test(self,batch, batch_idx, ways, shots, queries, method='l2'):
        '''Currently, we only support two meta test methods here in additional to the default one.
        The first is L2-regularized linear classifer (i.e., LogisticRegression from sklearn).
        The second is to finetune all layers (we found it has worse performance than the above one).
        '''

        self.features.eval()

        data, labels = batch
        labels = labels.long()


        if method == 'l2':
            features = self.features(data)
            features = features.detach()
            if self.norm_test_features:
                # Normalize features during test
                # We find this can make the LogisticRegression fitting data more stably and converge faster
                features = normalize(features)
            query_features, query_labels, support_features, support_labels = \
                get_query_support(features, labels, ways, shots, queries, detach_features=True,
                                  query_support_split=True, to_numpy=True,
                                  flatten_labels=True)
            clf = LogisticRegression(penalty='l2',
                                                 random_state=0,
                                                 C=self.test_l2, # C is the inverse of L2 regularization
                                                 solver='lbfgs',
                                                 max_iter=1000,
                                                 multi_class='auto')
            clf.fit(support_features, support_labels)
            query_preds= clf.predict(query_features)
            query_prob = clf.predict_proba(query_features)
        elif method == 'tune-all':
            # Finetuning all layers following Dhillon et al. ICLR'20
            # Note: Have to normalize all features
            assert self.test_phase
            query_data, query_labels, support_data, support_labels = \
                get_query_support(data, labels, ways, shots, queries, detach_features=False,
                                  query_support_split=True, to_numpy=False,
                                  flatten_labels=True)

            support_features = normalize(self.features(support_data).detach())
            X,y = support_features.detach().cpu().numpy(), support_labels.detach().cpu().numpy()
            centroids = NearestCentroid().fit(X,y).centroids_

            classifier = torch.nn.Linear(support_features.shape[1],ways)
            classifier.bias.data *= 0
            classifier.weight.data = torch.from_numpy(centroids).float()
            classifier.to(self.device)

            # Record parameters before adaptation
            if not hasattr(self,'test_state_dict'):
                self.test_state_dict = deepcopy(self.features.state_dict())
                for k,v in self.test_state_dict.items():
                    self.test_state_dict[k] = v.detach().data

            optim = torch.optim.Adam([{'params': self.features.parameters()},
                                      {'params': classifier.parameters(), "weight_decay": self.test_weight_decay}],
                                                                   lr=self.test_lr)
            for i in range(self.test_steps):
                optim.zero_grad()
                support_features = normalize(self.features(support_data))
                loss = self.loss(classifier(support_features), support_labels)
                loss.backward(retain_graph=True)
                optim.step()
            optim.zero_grad()
            with torch.no_grad():
                query_features = normalize(self.features(query_data)).detach()
                query_outputs = classifier(query_features).detach()
            query_outputs,query_labels = tensors2arrays(query_outputs,query_labels)
            query_preds = np.argmax(query_outputs,axis=1).flatten()
            query_prob = softmax(query_outputs,axis=1)
            # Reload parameters before the adaptation
            self.features.load_state_dict(self.test_state_dict)

        # Compute accuracy and loss
        acc = metrics.accuracy_score(query_labels, query_preds)
        loss = metrics.log_loss(query_labels, query_prob)
        return loss, acc
    def configure_optimizers(self):
        '''The available optimizers include SGD (with variants) and some popular adaptive optimization methods'''
        print('===============================')
        if self.optim[:3] == 'sgd':
            # SGD and its variants
            if self.optim == 'sgd':
                print(f'Using SGD w/ LR = {self.lr}, Momentum = {self.momentum}, Decay Rate = {self.decay_rate}, Decay Epochs = {self.decay_epochs}, weight decay = {self.weight_decay}')
                optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,momentum=self.momentum)
            elif self.optim == 'sgdp':
                print(f'Using SGDP w/ LR = {self.lr}, Momentum = {self.momentum}, Decay Rate = {self.decay_rate}, Decay Epochs = {self.decay_epochs}, weight decay = {self.weight_decay}')
                optimizer = torch_optimizer.SGDP(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,momentum=self.momentum)
            elif self.optim == 'sgdw':
                print(f'Using SGDW w/ LR = {self.lr}, Momentum = {self.momentum}, Decay Rate = {self.decay_rate}, Decay Epochs = {self.decay_epochs}, weight decay = {self.weight_decay}')
                optimizer = torch_optimizer.SGDW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,momentum=self.momentum)
            else: raise ValueError()

        else:
            # Adaptive optimization methods
            print(f'Using {self.optim} w/ LR = {self.lr}, weight decay = {self.weight_decay}, Decay Rate = {self.decay_rate}, Decay Epochs = {self.decay_epochs}')
            if self.optim == 'adam':
                optimizer = optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
            elif self.optim == 'adamw':
                optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
            elif self.optim == 'adamax':
                optimizer = optim.Adamax(self.parameters(), lr=2*self.lr, weight_decay=self.weight_decay)
            elif self.optim == 'radam':
                optimizer = torch_optimizer.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optim == 'adabound':
                optimizer = torch_optimizer.AdaBound(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optim == 'adamp':
                optimizer = torch_optimizer.AdamP(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else: raise ValueError()


        # We define the learnign rate scheduler below
        if self.scheduler == 'plateau':
            return {
                'optimizer': optimizer,
                'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    optimizer, mode='max',factor=self.decay_rate,patience=self.lr_decay_patience),
                'monitor': 'valid_accuracy'
                }
        elif self.scheduler == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step,
                gamma=self.scheduler_decay,
            )
            return [optimizer], [lr_scheduler]
        else:
            raise ValueError()
