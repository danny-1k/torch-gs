import torch
from torch.utils.data import DataLoader
from .metrics import Metric,Loss
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
        if params.get('net') == None:
            raise ValueError('params["net"] must not be None.')
        if (params.get('lossfn') == None) and (params.get(
            'criterion') == None):
            raise ValueError('params["lossfn"] or params["criterion"] must not be None.')
        if params.get(
            'optimizer') == None:
            raise ValueError('params["optimizer"] must not be None.')

        if not isinstance(params.get(
            'net'), torch.nn.Module):
            raise ValueError('params["net"] must be an instance of the `torch.nn.Module` class.')
        if not isinstance(params.get('lossfn'), torch.nn.Module) or isinstance(params.get(
            'criterion'), torch.nn.Module):
            raise ValueError('params["lossfn"] or params["criterion"] must be an instance of the `torch.nn.Module` class.')
        if not isinstance(params.get(
            'optimizer'), Optimizer):
            raise ValueError('params["optimizer"] must be an instance of the `Optimizer` class.')
        if not (isinstance(params.get('metric'), Metric) or params.get(
            'metric') == None):
            raise ValueError('params["metric"] must an instance of the `metrics.Metric` class.')

        self.net = params.get('net') or self.net
        self.lossfn = params.get('lossfn') or params.get(
            'criterion') or self.lossfn
        self.optimizer = params.get('optimizer') or self.optimizer
        self.lrschedulers = (([params.get('lrschedulers')] if type(params.get(
            'lrschedulers')) != list else params.get('lrschedulers'))if params.get('lrschedulers') != None else []) or self.lrschedulers if 'lrschedulers' in dir(self) else []
        self.metric = params.get(
            'metric') or Loss(self.lossfn)
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
        if not isinstance(
            trainloader, DataLoader):
                raise ValueError('`trainloader` must be a torch DataLoader')

        if testloader != None:
            if not isinstance(
                testloader, DataLoader):
                raise ValueError('`testloader` must be a torch DataLoader')

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

            self.net.eval()
            if testloader:
                self.performance[epoch + 1] =\
                    self.metric.evaluate(self.net, testloader)

            else:
                 self.performance[epoch + 1] =\
                    self.metric.evaluate(self.net, trainloader)


        return self.metric.get_performance(self.performance)


    def __repr__(self):
        return f'Trainer(metric={self.metric},optimizer={self.optimizer},)'