import argparse
import torch
from models import DenseLayerWithComplexNeurons, Conv2dLayerWithComplexNeurons


parser = argparse.ArgumentParser(description='Test layers with complex neurons')
parser.add_argument('--arity', default=2, type=int, help='arity of inner nets')


def test_dense(arity):
    inner_net_1 = torch.nn.Sequential(
        torch.nn.Linear(arity, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    inner_net_2 = torch.nn.Sequential(
        torch.nn.Linear(arity, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    inner_net_3 = torch.nn.Sequential(
        torch.nn.Linear(arity, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    outer_net = DenseLayerWithComplexNeurons([inner_net_1, inner_net_2, inner_net_3], arity, 16, 9)
    input = torch.randn(128, 16)
    output = outer_net(input)

    print(output.size())


def test_conv(arity):
    inner_net_1 = torch.nn.Sequential(
        torch.nn.Conv2d(arity, 10, (1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(10, 10, (1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(10, 1, (1, 1))
    )

    inner_net_2 = torch.nn.Sequential(
        torch.nn.Conv2d(arity, 10, (1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(10, 10, (1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(10, 1, (1, 1))
    )

    inner_net_3 = torch.nn.Sequential(
        torch.nn.Conv2d(arity, 10, (1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(10, 10, (1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(10, 1, (1, 1))
    )

    outer_net = Conv2dLayerWithComplexNeurons([inner_net_1, inner_net_2, inner_net_3], arity, 3, 9, (3, 3))
    input = torch.randn(128, 3, 32, 32) 
    output = outer_net(input)

    print(output.size())


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    test_conv(args.arity)
    
