# -*- coding: utf-8 -*-

"""Console script for private_deep_learning."""

import sys
import os
import click

import torch
import torch.nn.functional
import syft
import torchvision
import numpy

# local modules
from mjpegc import mjpegc

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


def do_predict(model_filename) -> int:

    #TODO: refactor and resolve an issue with feeding jpegs to neural net

    """
    from PIL import Image
    image = Image.open("kissa.jpg")
    image = numpy.array(image)  # convert image to numpy array
    t = torch.from_numpy(image)
    t = t.unsqueeze(0)
    model.forward(t)
    k = model(t)
    result = os.EX_OK
    """
    return os.EX_OK


def do_generate(count, batch_size, test_batch_size, device, epochs, log_interval, model_filename) -> int:
    """
    generates and saves a neural network model with federated workers
    parameters can be used to guide training
    :param count: number of federated workers
    :param batch_size: size of batch to a federated worker
    :param test_batch_size: test batch size
    :param device: cpu (tested) or cuda (not tested)
    :param epochs: neural network training epochs
    :param log_interval: interval to print training progress and accuracy
    :param model_filename: filename to save the trained model to
    :return: os.EX_OK on success
    """
    hook = syft.TorchHook(torch)
    workers = get_workers(count, hook)
    fdl = get_federated_train_loader(workers, batch_size)
    test_loader = get_test_loader(test_batch_size)
    model = Net().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        train(model, device, fdl, optimizer, epoch, log_interval, batch_size)
        test(model, device, test_loader)

    torch.save(model.state_dict(), model_filename)
    return os.EX_OK


def read_stream(url):
    """
    read MJPEG stream from url
    :param url: http url to the server stream
    :return: yields frames to generator
    """

    # TODO: remove visual cv2 debug, yield frame to feed to neural net instead
    import cv2

    for jpg in mjpegc.client(url):
        i = cv2.imdecode(numpy.fromstring(jpg, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('mjpeg stream', i)
        if cv2.waitKey(1) == 27: # esc-key
            break
    return os.EX_OK

@click.command()
@click.option('--count', default=2, help='Number of workers')
@click.option('--batch_size', default=64, help='Batch size to share to federated workers')
@click.option('--test_batch_size', default=1000, help='Test batch size')
@click.option('--device', default="cpu", help='torch device to use: ["cpu"|"cuda"]')
@click.option('--log_interval', default=200)
@click.option('--epochs', default=20)
@click.option('--model_filename', default="federated_trained_cifar10_model.pt")
@click.option('--predict/-no-predict', default=False)
@click.option('--generate/--no-generate', default=False)
@click.option('--stream/--no_stream', default=True)
@click.option('--stream_url', default="http://localhost:5000", help='url to HTTP MJPEG stream')
def main(count, batch_size,  test_batch_size, device,
         log_interval, epochs, model_filename, predict, generate, stream, stream_url) -> int:

    result = os.EX_UNAVAILABLE

    try:
        if stream: result = read_stream(stream_url)
        if predict: result = do_predict(model_filename)
        if generate: result = do_generate(count, batch_size, test_batch_size, device, epochs, log_interval, model_filename)
    except:
        print("Unexpected error:", sys.exc_info())
        result = os.EX_SOFTWARE

    return result


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
