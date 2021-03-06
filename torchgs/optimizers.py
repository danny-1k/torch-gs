
class Optimizer:
    def __init__(self, optimizer: type, net, params: dict):
        """
        Optimizer class

        Args:
            optimizer (type): optimizer
            params (dict): dictionary of parameters
        """
        self.params = params
        self.net = net
        self.optimizer = optimizer(net.parameters(), **params or {})


    def __repr__(self):
        return f'Optimizer({type(self.optimizer).__name__})'


class LRscheduler:
    def __init__(self, lrscheduler: type, params: dict):
        """
        Lrscheduler class

        Args:
            lrscheduler (type): lrscheduler
            params (dict): dictionary of parameters
        """
        self.params = params
        self.lrscheduler = lrscheduler(**params)

    
    def __repr__(self):
        return f'LRscheduler({type(self.lrscheduler).__name__})'