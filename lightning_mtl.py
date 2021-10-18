import numpy as np
import torch
from learn2learn.utils import accuracy
from lightning_episodic_module import LightningEpisodicModule,get_query_support
from torch.nn import functional as F
import math


class LightningMTL(LightningEpisodicModule):

    """

    **Description**

    A PyTorch Lightning module for multi-task learning.

    **Arguments**

    * **features** (Module) - A nn.Module to extract features, which will not be adaptated.
    * **classifier** (Module) - A nn.Module taking features, mapping them to classification.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
    * **ways** (int, *optional*, default=5) - Number of classes in a task.
    * **shots** (int, *optional*, default=1) - Number of samples for adaptation.
    * **adaptation_steps** (int, *optional*, default=1) - Number of steps for adapting to new task.
    * **lr** (float, *optional*, default=0.001) - Learning rate of meta training.
    * **adaptation_lr** (float, *optional*, default=0.1) - Learning rate for fast adaptation.
    * **scheduler_step** (int, *optional*, default=20) - Decay interval for `lr`.
    * **scheduler_decay** (float, *optional*, default=1.0) - Decay rate for `lr`.

    **References**

    Wang et al. ICML 2021. "Bridging Multi-Task Learning and Meta-Learning: Towards Efficient Training and Effective Adaptation"

    """
    adaptation_steps = 50
    adaptation_lr = 0.01
    test_adaptation_steps = 50
    def __init__(self, features, classifier, loss=None, **kwargs):
        super().__init__(**kwargs)
        if loss is None:
            loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.loss = loss
        self.train_ways = kwargs.get("train_ways", LightningEpisodicModule.train_ways)
        self.train_shots = kwargs.get(
            "train_shots", LightningEpisodicModule.train_shots
        )
        self.train_queries = kwargs.get(
            "train_queries", LightningEpisodicModule.train_queries
        )
        self.test_ways = kwargs.get("test_ways", LightningEpisodicModule.test_ways)
        self.test_shots = kwargs.get("test_shots", LightningEpisodicModule.test_shots)
        self.test_queries = kwargs.get(
            "test_queries", LightningEpisodicModule.test_queries
        )
        self.lr = kwargs.get("lr", LightningEpisodicModule.lr)
        self.scheduler_step = kwargs.get(
            "scheduler_step", LightningEpisodicModule.scheduler_step
        )
        self.scheduler_decay = kwargs.get(
            "scheduler_decay", LightningEpisodicModule.scheduler_decay
        )
        self.adaptation_steps = kwargs.get(
            "adaptation_steps", LightningMTL.adaptation_steps
        )
        self.test_adaptation_steps = kwargs.get(
            "test_adaptation_steps", LightningMTL.test_adaptation_steps
        )
        self.adaptation_lr = kwargs.get("adaptation_lr", LightningMTL.adaptation_lr)
        self.data_parallel = kwargs.get("data_parallel", False)
        self.test_method = kwargs.get("test_method", 'default')
        self.finetune_steps = kwargs.get("finetune_steps", 50)

        self.features = features
        if self.data_parallel and torch.cuda.device_count() > 1:
            self.features = torch.nn.DataParallel(self.features)
        self.classifier = classifier
        self.hyperparams.update({
            "train_ways": self.train_ways,
            "train_shots": self.train_shots,
            "train_queries": self.train_queries,
            "test_ways": self.test_ways,
            "test_shots": self.test_shots,
            "test_queries": self.test_queries,
            "lr": self.lr,
            "scheduler_step": self.scheduler_step,
            "scheduler_decay": self.scheduler_decay,
            "adaptation_lr": self.adaptation_lr,
            "adaptation_steps": self.adaptation_steps,
            "test_method": self.test_method,
            "finetune_steps": self.finetune_steps,
            "test_adaptation_steps": self.test_adaptation_steps,
        })
        self.save_hyperparameters(self.hyperparams)

        self.map_train_labels = False

        # Our multi-task learning implementation needs to know the number of training classes
        # to construct heads
        if kwargs["dataset"] in ['cifarfs','mini-imagenet']:
            self.n_train_class = 64

        elif kwargs['dataset'] == 'tiered-imagenet':
            self.n_train_class = 351
        elif kwargs['dataset'] == 'fc100':
            #  The learn2learn dataloader of FC100 doesn't preserve the original labels of training data,
            #  thus we have to manually map the remapped labels (i.e., 0,1,...,59) to the original labels below.
            self.n_train_class = 60
            orig_labels = np.array([0, 1, 5, 8, 9, 10, 12, 13, 16, 17, 20, 22, 23, 25, 27, 28, 29,
                                    32, 33, 37, 39, 40, 41, 44, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58,
                                    59, 60, 61, 62, 67, 68, 69, 70, 71, 73, 76, 78, 81, 82, 83, 84, 85,
                                    86, 87, 89, 90, 91, 92, 93, 94, 96])
            label_mapping = np.zeros(100, dtype=int)
            label_mapping[orig_labels] = np.arange(60)
            self.label_mapping = torch.from_numpy(label_mapping).long()
            self.map_train_labels = True

        self.embedding_size = classifier.weight.shape[1]
        self.output_size = classifier.weight.shape[0]
        train_heads = torch.Tensor(self.n_train_class,self.embedding_size)
        torch.nn.init.kaiming_uniform_(train_heads,a=math.sqrt(5))
        train_heads = train_heads*math.sqrt(self.n_train_class/self.train_ways) # Compensate for the kaiming_init for larger dim
        self.train_heads = torch.nn.Parameter(train_heads)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningEpisodicModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--adaptation_steps",
            type=int,
            default=LightningMTL.adaptation_steps,
        )
        parser.add_argument(
            "--adaptation_lr",
            type=float,
            default=LightningMTL.adaptation_lr,
        )
        parser.add_argument(
            "--data_parallel",
            action='store_true',
            help='Use this + CUDA_VISIBLE_DEVICES to parallelize across GPUs.',
        )
        parser.add_argument(
            "--test-adaptation_steps",
            type=int,
            default=LightningMTL.test_adaptation_steps,
        )
        parser.add_argument(
            "--head-weight-decay",
            type=float,
            default=-1,
        )
        parser.add_argument(
            "--test-head-weight-decay",
            type=float,
            default=0.1,
        )

        parser.add_argument(
            "--split",
            default=False,
            action="store_true"
        )


        return parser


    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, ways, shots, queries):

        # For MTL training, we do not split the batch into query and support data. See our paper for details
        # During test, we have to split the batch into query and support data, since the model finetunes
        # its last layer on the labelled support data and predict on the unlabeled query data.

        # ERM training
        self.features.train()
        data, labels = batch
        labels = labels.long()
        if self.map_train_labels:
            self.label_mapping = self.label_mapping.to(labels.device)
            labels = self.label_mapping[labels]
        class_idxes = torch.unique_consecutive(labels)  # Do not sort the labels/idxes
        weight = self.train_heads[class_idxes]
        assert weight.shape == self.classifier.weight.shape
        def learner(x):
            return F.linear(x, weight)
        labels_new = torch.zeros_like(labels).long()
        for i, class_idx in enumerate(class_idxes):            
            labels_new[labels == class_idx] = i        
        preds = learner(self.features(data))
        error = self.loss(preds, labels_new)
        acc = accuracy(preds, labels_new)
        return error,acc
