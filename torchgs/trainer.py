import torch
from torch.utils.data import DataLoader
from .metrics import Metric
from .optimizers import Optimizer


class Trainer:
    def __init__(self, params: dict):
        """
        Trainer class for training a pytorch model

        Args:
            params (dict): dictionary containing objects necessary to 
            train the network
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._set_params(params)

    def _set_params(self, params: dict = {}):
        """
        Sets the parameters of the Trainer to the supplied

        Args:
            params (dict, optional): dictionary containing objects
            necessary to train the network. Defaults to {}.
        """
        assert params.get('net') != None, 'params["net"] must not be None.'
        assert params.get('lossfn') != None or params.get(
            'criterion') != None, 'params["lossfn"] or params["criterion"] must not be None.'
        assert params.get(
            'optimizer') != None, 'params["optimizer"] must not be None.'

        assert isinstance(params.get(
            'net'), torch.nn.Module), 'params["net"] must be an instance of the `torch.nn.Module` class.'
        assert isinstance(params.get('lossfn'), torch.nn.Module) or isinstance(params.get(
            'criterion'), torch.nn.Module), 'params["lossfn"] or params["criterion"] must be an instance of the `torch.nn.Module` class.'
        assert isinstance(params.get(
            'optimizer'), Optimizer), 'params["optimizer"] must be an instance of the `Optimizer` class.'
        assert isinstance(params.get('metric'), Metric) or params.get(
            'metric') == None, 'params["metric"] must an instance of the `metrics.Metric` class.'

        self.net = params.get('net') or self.net
        self.lossfn = params.get('lossfn') or params.get(
            'criterion') or self.lossfn
        self.optimizer = params.get('optimizer') or self.optimizer
        self.lrschedulers = (([params.get('lrschedulers')] if type(params.get(
            'lrschedulers')) != list else params.get('lrschedulers'))if params.get('lrschedulers') != None else []) or self.lrschedulers if 'lrschedulers' in dir(self) else []
        self.metric = params.get(
            'metric') or self.metric if 'metric' in dir(self) else None
        self.performance = {}
        self.net.to(self.device)

    def train_on(self, trainloader: DataLoader, testloader: DataLoader = None, epochs: int = 1):
        """
        Method to train the network

        Args:
            trainloader (DataLoader): Train dataloader for training the model.
            testloader (DataLoader, optional): Test dataloader for testing the models performance. Defaults to None.
            epochs (int, optional): Number of epochs to train the network for. Defaults to 1.
        """
        assert isinstance(
            trainloader, DataLoader), '`trainloader` must be a torch DataLoader'
        self.performance['train'] = {}

        if testloader != None:
            assert isinstance(
                testloader, DataLoader), '`testloader` must be a torch DataLoader'
            self.performance['test'] = {}

        for epoch in range(epochs):

            self.net.train()
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.optimizer.zero_grad()

                p = self.net(x)
                loss = self.lossfn(p, y)
                loss.backward()
                self.optimizer.optimizer.step()

            for sched in self.lrschedulers:
                sched.step(loss)

            if self.metric:
                self.performance['train'][epoch +
                                          1] = self.metric.evaluate(self.net, trainloader)
            else:
                self.performance['train'][epoch+1] = loss.item()

            self.net.eval()
            with torch.no_grad():

                if testloader:

                    if self.metric:
                        self.performance['test'][epoch +
                                                 1] = self.metric.evaluate(self.net, testloader)

                    else:

                        for x, y in testloader:
                            x = x.to(self.device)
                            y = y.to(self.device)

                            p = self.net(x)
                            loss = self.lossfn(p, y)

                        self.performance['test'][epoch+1] = loss.item()

        return self.performance
