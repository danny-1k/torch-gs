import torch

class Optimizer:
    def __init__(self,optimizer:type,params:dict):
        """
        Optimizer class

        Args:
            optimizer (type): optimizer
            params (dict): dictionary of parameters
        """        
        self.params = params
        self.optimizer = optimizer(*params)


class LRscheduler:
    def __init__(self,lrscheduler:type,params:dict):
        """
        Lrscheduler class

        Args:
            lrscheduler (type): lrscheduler
            params (dict): dictionary of parameters
        """        
        self.params = params
        self.lrscheduler = lrscheduler(*params)
