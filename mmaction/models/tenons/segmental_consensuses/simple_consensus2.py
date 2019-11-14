import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES2

class _SimpleConsensus2(torch.autograd.Function):
    """Simplest segmental consensus module"""

    def __init__(self,
                 consensus_type='avg',
                 dim=1):
        super(_SimpleConsensus2, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, x):
        self.shape = x.size()
        if self.consensus_type == 'avg':
            output = x.mean(dim=self.dim, keepdim=True)
        else:
            output = None
        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        else:
            grad_in = None
        return grad_in


@SEGMENTAL_CONSENSUSES2.register_module
class SimpleConsensus2(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus2, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        return _SimpleConsensus2(self.consensus_type, self.dim)(input)