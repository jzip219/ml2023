import torch
from torch.autograd import Variable

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, inputSize=1, outputSize=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        return self.linear(x)

    

""" class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.linear(x) """