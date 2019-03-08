# -*- coding: utf-8 -*-

"""Console script for private_deep_learning."""

import sys
import os
import click

import torch
import torch.nn.functional
import syft
import torchvision


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def get_workers(count, hook) -> tuple:
    return tuple([syft.VirtualWorker(hook, id=str(i)) for i in range(count)])


def get_federated_train_loader(workers, batch_size) -> syft.FederatedDataLoader:
    return syft.FederatedDataLoader(
        # using CIFAR10 as one of the classes is _cats_
        torchvision.datasets.CIFAR10('../data',
                               train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])).federate(workers), batch_size=batch_size, shuffle=True)


def train(model, device, train_loader, optimizer, epoch, log_interval, batch_size):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader): # <-- now it is a distributed dataset
        model.send(data.location) # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get() # <-- NEW: get the model back

        if batch_idx % log_interval == 0:
            loss = loss.get() # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader) * batch_size,
                100. * batch_idx / len(train_loader), loss.item()))


def get_test_loader(test_batch_size) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('../data', train=False, transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])), batch_size=test_batch_size, shuffle=True)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


@click.command()
@click.option('--count', default=2, help='Number of workers')
@click.option('--batch_size', default=64, help='Batch size to share to federated workers')
@click.option('--test_batch_size', default=1000, help='Test batch size')
@click.option('--device', default="cpu", help='torch device to use: ["cpu"|"cuda"]')
@click.option('--log_interval', default=200)
@click.option('--epochs', default=20)
@click.option('--model_filename', default="federated_trained_cifar10_model.pt")
def main(count, batch_size,  test_batch_size, device,
         log_interval, epochs, model_filename) -> int:

    result = os.EX_CANTCREAT

    try:
        hook = syft.TorchHook(torch)
        workers = get_workers(count, hook)
        fdl = get_federated_train_loader(workers, batch_size)
        test_loader = get_test_loader(test_batch_size)
        model = Net().to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(1, epochs+1):
            train(model, device, fdl, optimizer, epoch, log_interval, batch_size)
            test(model, device, test_loader)

        torch.save(model.state_dict(), model_filename)
        result = os.EX_OK
    except:
        result = os.EX_SOFTWARE

    return result


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
