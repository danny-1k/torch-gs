import copy
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from .trainer import Trainer

from .optimizers import Optimizer
from .metrics import Loss

class GridSearch:
    def __init__(self, param_space, trainer):
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
        self.search_space = self.gen_combinations(self.search_space) # generate all possible combinations of combinations
        self.trainer = trainer
        # self.original_trainer = copy.deepcopy(self.trainer)
        # self.original_net = copy.deepcopy(self.trainer.net)

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
        assert isinstance(
            trainset, Dataset), '`trainset` must be a `torch.utils.data.Dataset` instance'
        assert isinstance(
            testset, Dataset) or testset == None, '`trainset` must be a `torch.utils.data.Dataset` instance'

        trainer_params = self.search_space.get('trainer')
        optimizer_params = self.search_space.get('optimizer')
        lrscheduler_params = self.search_space.get('lrcheduler')

    def fit_once(self,net,params):
        trainer_params = params.get('trainer')
        optimizer_params = params.get('optimizer')
        lrscheduler_params = params.get('lrcheduler')


        optimizer = Optimizer(trainer_params.get('optimizer'),optimizer_params)

        trainer = Trainer(
            {
                'net':net,
                'lossfn':trainer_params.get('lossfn'),
                'optimizer':optimizer,
                'lrschedulers':trainer_params.get('lrcheduler'),
                'metric':trainer_params.get('metric'),
            }
        )



