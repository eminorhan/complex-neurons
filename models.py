import math
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module


class DenseLayerWithComplexNeurons(Module):
    r"""Applies a dense layer with complex neurons to incoming data. It can be used in the same way as ``nn.Linear``.

    Args:
        cell_types: list of inner nets (can be arbitrary modules)
        arity: common arity of inner nets
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """
    __constants__ = ['arity', 'in_features', 'out_features']
    arity: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,  cell_types, arity: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(DenseLayerWithComplexNeurons, self).__init__()
        self.cell_types = cell_types
        self.arity = arity
        self.num_cell_types = len(cell_types)
        self.in_features = in_features
        self.out_features = out_features

        assert self.out_features % self.num_cell_types == 0  # make sure each cell type has the same # of neurons
        self.cell_indices = np.repeat(range(self.num_cell_types), self.out_features // self.num_cell_types)

        self.weight = Parameter(torch.Tensor(arity * out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(arity * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        all_inputs = F.linear(input, self.weight, self.bias)
        all_outputs = []
        for i in range(self.out_features):
            o = self.cell_types[self.cell_indices[i]](all_inputs[..., i*self.arity:(i+1)*self.arity])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, -1)

        return all_outputs

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features,
                                                                 self.bias is not None)