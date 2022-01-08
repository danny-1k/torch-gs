import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score,recall_score,f1_score


class Metric:
    def __init__(self):
        pass

    def evaluate(self, net: nn.Module, loader: DataLoader):
        return self._evaluate_func(net, loader)

    def get_performance(self, performance: dict):
        vals = torch.Tensor(list(performance.values()))

        min_val = torch.min(vals).item()
        max_val = torch.mean(vals).item()
        mean_val = torch.mean(vals).item()
        std_val = torch.std(vals).item()
        last = vals[-1].item()

        return {'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'last': last,
                }

    
    def __repr__(self):
        return f'Metric({type(self).__name__})'


class Loss(Metric):
    def __init__(self, lossfn: nn.Module):
        super().__init__()
        self.lossfn = lossfn
        self.maximize = False

    def _evaluate_func(self, net: nn.Module, loader: DataLoader):
        with torch.no_grad():
            mean_loss = []

            for x, y in loader:
                loss = self.lossfn(net(x), y)
                mean_loss.append(loss.item())

            mean_loss = sum(mean_loss)/len(mean_loss)

        return mean_loss


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def _evaluate_func(self, net: nn.Module, loader: DataLoader):
        with torch.no_grad():
            accuracy = 0
            for x, y in loader:
                pred = net(x)
                accuracy += sum(pred.argmax(dim=1) == y).item()
            accuracy /= len(loader.dataset)

        return accuracy


class Recall(Metric):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def _evaluate_func(self, net: nn.Module, loader: DataLoader):
        with torch.no_grad():
            recall = 0
            for x, y in loader:
                pred = net(x)
                recall += recall_score(y,pred.argmax(dim=1),average='micro').item()

            recall /= len(loader.dataset)

        return recall



class Precision(Metric):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def _evaluate_func(self, net: nn.Module, loader: DataLoader):
        with torch.no_grad():
            precision = 0
            for x, y in loader:
                pred = net(x)
                precision += precision_score(y,pred.argmax(dim=1),average='micro').item()

            precision /= len(loader.dataset)

        return precision


class F1(Metric):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def _evaluate_func(self, net: nn.Module, loader: DataLoader):
        with torch.no_grad():
            f1 = 0
            for x, y in loader:
                pred = net(x)
                f1 += precision_score(y,pred.argmax(dim=1),average='micro').item()

            f1 /= len(loader.dataset)

        return f1