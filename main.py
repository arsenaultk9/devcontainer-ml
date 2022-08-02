# Code derived from original example: https://github.com/pytorch/examples/tree/main/mnist

from __future__ import print_function
# import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from args import Args
from network import Network
from network_trainor import train, test

def main(args):
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.MNIST('../data', train=True, download=True,
                                   transform=transform)
    dataset_test = datasets.MNIST('../data', train=False,
                                  transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = Network().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    args = Args()
    main(args)
