import torch
from .metrics import Metric


class Trainer:
    def __init__(self, params):
        """
        Trainer class for training a pytorch model

        Args:
            params (dict): dictionary containing objects necessary to 
            train the network
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._set_params(params)

    def _set_params(self, params={}):
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
            'optimizer'), torch.optim.Optimizer), 'params["optimizer"] must be an instance of the `torch.optim.Optimizer` class.'
        assert isinstance(params.get('metric'), Metric) or params.get(
            'metric') == None, 'params["metric"] must an instance of the `metrics.Metric` class.'

        self.net = params.get('net')
        self.lossfn = params.get('lossfn') or params.get('criterion')
        self.optimizer = params.get('optimizer')
        self.lrschedulers = ([params.get('lrschedulers')] if type(params.get(
            'lrschedulers')) != list else params.get('lrschedulers'))if params.get('lrschedulers') != None else []
        self.metric = params.get('metric')
        self.performance = {}

        self.net.to(self.device)

    def train_on(self, trainloader, testloader=None, epochs=1):
        """
        Method to train the network

        Args:
            trainloader (torch.utils.data.DataLoader): Train dataloader for training the model.
            testloader (torch.utils.data.DataLoader, optional): Test dataloader for testing the models performance. Defaults to None.
            epochs (int, optional): Number of epochs to train the network for. Defaults to 1.
        """
        assert isinstance(
            trainloader, torch.utils.data.DataLoader), '`trainloader` must be a torch DataLoader'
        self.performance['train'] = {}

        if testloader != None:
            assert isinstance(
                testloader, torch.utils.data.DataLoader), '`testloader` must be a torch DataLoader'
            self.performance['test'] = {}

        for epoch in range(epochs):

            self.net.train()
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                p = self.net(x)
                loss = self.lossfn(p, y)
                loss.backward()
                self.optimizer.step()

            for sched in self.lrschedulers:
                sched.step(loss)

            self.performance['train'][epoch+1] = loss.item()

            self.net.eval()
            with torch.no_grad():

                if testloader:

                    for x, y in testloader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        p = self.net(x)
                        loss = self.lossfn(p, y)

                    self.performance['test'][epoch+1] = loss.item()

        return self.performance
