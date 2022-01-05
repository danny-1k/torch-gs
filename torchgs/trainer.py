import torch
from .metrics import Metric

class Trainer:
    def __init__(self,net,lossfn,optimizer,lrschedulers=None,losshandler=None,metric=None):
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net
        self.lossfn = lossfn
        self.optimizer = optimizer
        self.losshandler = losshandler
        self.lrschedulers = lrschedulers if lrschedulers!=None else []
        self.performance = {}
        self.metric = metric

        self.net.to(self.device)

        if self.metric !=None:
            assert isinstance(metric,Metric), '`metric` must be a class inherited from `metrics.Metric`'


    def train_on(self,trainloader,testloader=None,epochs=1):
        assert isinstance(trainloader,torch.utils.data.DataLoader) , '`trainloader` must be a torch DataLoader'
        self.performance['train'] = {}
        
        if testloader!=None:
            assert isinstance(testloader,torch.utils.data.DataLoader) , '`testloader` must be a torch DataLoader'
            self.performance['test'] = {}


        for epoch in range(epochs):
            
            self.net.train()
            for x,y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                p = self.net(x)
                loss = self.lossfn(p,y)
                loss.backward()
                self.optimizer.step()

                for sched in self.lrschedulers:
                    sched.step()

            self.performance['train'][epoch+1] = loss.item()


            self.net.eval()
            with torch.no_grad():

                if testloader:

                    for x,y in testloader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        p = self.net(x)
                        loss = self.lossfn(p,y)

                    self.performance['test'][epoch+1] = loss.item()
            
        return self.performance