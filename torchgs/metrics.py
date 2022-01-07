import torch
import torch.nn as nn

class Metric:
    def __init__(self):
        pass

    def evaluate(self,net,loader):
        return self._evaluate_func(net,loader)


class Loss(Metric):
    def __init__(self,lossfn:nn.Module):
        super().__init__()
        self.lossfn = lossfn
        self.maximize = False

    def _evaluate_func(self,net,loader):
        with torch.no_grad:
            mean_loss = []

            for x,y in loader:
                loss = self.lossfn(net(x),y)
                mean_loss.append(loss.item())

            mean_loss = sum(mean_loss)/len(mean_loss)

        return mean_loss






class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def _evaluate_func(self,net,loader):
        with torch.no_grad:
            accuracy = 0
            for x,y in loader:
                pred = net(x)
                accuracy+=sum(pred.argmax(dim=1)==y)
            accuracy = accuracy/len(loader.dataset)


class Recall(Metric):
    def __init__(self):
        super().__init__()
        self.maximize = True


class Precision(Metric):
    def __init__(self):
        super().__init__()
        self.maximize=True


class F1(Metric):
    def __init__(self):
        super().__init__()
        self.maximize = True
