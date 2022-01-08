import copy
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from .trainer import Trainer

from .optimizers import Optimizer, LRscheduler
from .metrics import Loss


class GridSearch:
    def __init__(self, param_space: dict, net: nn.Module=None):
        """
        Grid search class for finding the optimal 
        set of hyperparameters

        Args:
            param_space (dict): Hyperparameters to search
            net (torch.nn.Module): Network to optimize

        """
        self.param_space = param_space
        self.search_space = {space: self.gen_combinations(
            self.param_space[space]) for space in self.param_space}  # generate all possible combinations
        # of parameters for each group
        # generate all possible combinations of combinations
        self.search_space = self.gen_combinations(self.search_space)
        self.net = net

    def gen_combinations(self, param_space):
        """
        Function to generate all possible combinations of 
        parameters in a param_space
        """

        params = []

        if len(param_space) > 1:

            for idx in range(len(param_space)-1):

                if idx == 0:
                    current_key = list(param_space.keys())[idx]
                    next_key = list(param_space.keys())[idx+1]
                    current = param_space[current_key]
                    next = param_space[next_key]

                    for param_current in current:
                        for param_next in next:
                            params.append(
                                {current_key: param_current, next_key: param_next})

                else:
                    next_key = list(param_space.keys())[idx+1]
                    next = param_space[next_key]
                    next_params = []

                    for idx, param in enumerate(params):
                        for param_next in next:
                            orig_param = param.copy()
                            orig_param[next_key] = param_next

                            next_params.append(orig_param)

                    params = next_params
        else:
            for param in param_space:
                for item in param_space[param]:
                    params.append({param: item})

        return params

    def fit(self, trainset: Dataset, testset: Dataset = None):
        """
        Performs the grid search and returns the summary

        Args:
            trainset (Dataset): train dataset
            testset (Dataset, optional): test dataset. Defaults to None.
        """
        assert isinstance(
            trainset, Dataset), '`trainset` must be a `torch.utils.data.Dataset` instance'
        assert isinstance(
            testset, Dataset) or testset == None, '`trainset` must be a `torch.utils.data.Dataset` instance'

        results = {}

        for idx, hypothesis in enumerate(self.search_space):
            performance = self.fit_once(
                net=copy.deepcopy(self.net),
                params=hypothesis,
                train=trainset,
                test=testset,
            )

            results[idx] = {'parameter_set': hypothesis,
                            'performance': performance}

        return results

        # trainer_params = self.search_space.get('trainer')
        # optimizer_params = self.search_space.get('optimizer')
        # lrschedulers_params = self.search_space.get('lrchedulers')

    def fit_once(self, net, params, train, test):
        """
        Fit the net with a current set of parameters, a
        train dataset and an optional test dataset

        Args:
            net (nn.Module): network 
            params (dict): parameter dict
            train (Dataset): train dataset
            test (Dataset,optional): test dataset

        Returns:
            dict: performance over the epochs trained on
        """
        trainer_params = params.get('trainer')
        optimizer_params = params.get('optimizer')
        lrschedulers_params = params.get('lrchedulers')
        train_loader_params = params.get('train_loader')
        test_loader_params = params.get('test_loader')

        if type(trainer_params.get('lrchedulers')) == list:
            lrschedulers = []

            for idx, sched in enumerate(trainer_params.get('lrchedulers')):
                lrschedulers.append(LRscheduler(
                    sched, lrschedulers_params[idx]))

        else:
            lrschedulers = None

        optimizer = Optimizer(trainer_params.get(
            'optimizer'), trainer_params.get('net') or net, optimizer_params)

        trainer = Trainer(
            {
                'net': trainer_params.get('net') or net,
                'lossfn': copy.deepcopy(trainer_params.get('lossfn')),
                'optimizer': optimizer,
                'lrschedulers': lrschedulers,
                'metric': copy.deepcopy(trainer_params.get('metric')),
            }
        )

        train_loader = DataLoader(train, **train_loader_params)
        test_loader = DataLoader(test, **test_loader_params) if test else None

        return trainer.train_on(
            trainloader=train_loader,
            testloader=test_loader,
            epochs=trainer_params.get('epochs') or 1,
        )
