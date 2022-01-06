import torch

class Optimizer:
    def __init__(self,optimizer,params):
        """
        Optimizer class

        Args:
            optimizer (torch.optim.Optimizer): optimizer
            params (dict): dictionary of parameters
        """        
        self.optimizer = optimizer
        self.params = params
        self._set_params(self,self.params)

    def _set_params(self,params):
        """
        Method to change some attributes of the optimizer

        Args:
            params (dict): dict of attributes and values
        """        
        for param in params:
            self.optimizer.__setattr__(param,self.params[param])