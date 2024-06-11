import torch
from torch import nn
import math


class SharedBiasLinear(nn.Module):
    def __init__(self, num_inputs, num_units):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.linear = nn.Linear(num_inputs, num_units, bias=False)
        self.bias = nn.Parameter(data=torch.tensor(torch.zeros(1), dtype=self.linear.weight.dtype), 
                                 requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
        bound = math.sqrt(1 / fan_in)
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        linear = self.linear(input)
        bias = torch.broadcast_to(self.bias, linear.size())
        return linear + bias

    def extra_repr(self):
        return 'num_inputs={}, num_units={}'.format(
            self.num_inputs, self.num_units
        )
        

def haiku_initializer(self, m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        boundary = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(m.weight, -boundary, boundary)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -boundary, boundary)
