import argparse
import torch
from models import DenseLayerWithComplexNeurons


parser = argparse.ArgumentParser(description='Test layers with complex neurons')
parser.add_argument('--arity', default=2, type=int, help='arity of inner nets')


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

    inner_net_1 = torch.nn.Sequential(
        torch.nn.Linear(args.arity, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    inner_net_2 = torch.nn.Sequential(
        torch.nn.Linear(args.arity, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    inner_net_3 = torch.nn.Sequential(
        torch.nn.Linear(args.arity, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    outer_net = DenseLayerWithComplexNeurons([inner_net_1, inner_net_2, inner_net_3], args.arity, 16, 9)
    input = torch.randn(128, 16)
    output = outer_net(input)

    print(output.size())