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

import PIL
import io

# local modules
from mjpegc import mjpegc
from nnet import nnet

# CIFAR labels
model_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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


def do_predict(model_filename, url) -> int:

    model = nnet.Net()
    model.load_state_dict(torch.load(model_filename))

    # transform pipeline for our 640x480 mjpeg stream input jpeg images with random crop to square,
    # resize to cifar10 dataset sample size 32x32 and finally to normalized tensor
    transform_pipeline = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(480, 480),
                                                         torchvision.transforms.Resize((32, 32)),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                          (0.5, 0.5, 0.5))])
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 100)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    last_prediction = -1
    consecutive_match, consecutive_mismatch = 0, 0
    threshold = 3
    overlay_text = ""

    for jpg in mjpegc.client(url):
        # process jpeg through PIL and above transform pipeline
        # unsqueeze converts our tensor (n channels, height, width) to
        # pytorch model input tensor (n images, n channels, height, width)
        img = transform_pipeline(PIL.Image.open(io.BytesIO(jpg))).unsqueeze(0)
        # wrap input tensor to Variable for prediction
        img = torch.autograd.Variable(img)
        prediction = model(img)
        """
        # using classic cifar10 set trained model gives us much better than chance (1/10)
        # prediction at

        Train Epoch: 20 [0/50048 (0%)]	Loss: 0.858092
        Train Epoch: 20 [12800/50048 (26%)]	Loss: 1.153132
        Train Epoch: 20 [25600/50048 (51%)]	Loss: 1.031593
        Train Epoch: 20 [38400/50048 (77%)]	Loss: 0.982400

        Test set: Average loss: 1.1617, Accuracy: 5922/10000 (59%)

        # but still a lot of false positives at 30 frames and predictions per second
        # TODO: filter single predictions from consecutive to reduce noise
        """

        current_prediction = prediction.data.numpy().argmax()

        i = cv2.imdecode(numpy.fromstring(jpg, dtype=numpy.uint8), cv2.IMREAD_COLOR)

        if current_prediction == last_prediction:
            consecutive_match += 1
            consecutive_mismatch = 0
        else:
            consecutive_mismatch += 1
            consecutive_match = 0

        if consecutive_match >= threshold:
            overlay_text = "the cat is alive"
        elif consecutive_mismatch >= threshold:
            overlay_text = ""

        cv2.putText(i, overlay_text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        last_prediction = current_prediction

        cv2.imshow('schroedingers', i)
        if cv2.waitKey(1) == 27:  # esc-key
            break

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
    model = nnet.Net().to(device)

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
@click.option('--predict/-no-predict', default=True)
@click.option('--generate/--no-generate', default=False)
@click.option('--stream/--no_stream', default=False)
@click.option('--stream_url', default="http://localhost:5000", help='url to HTTP MJPEG stream')
def main(count, batch_size,  test_batch_size, device,
         log_interval, epochs, model_filename, predict, generate, stream, stream_url) -> int:

    result = os.EX_UNAVAILABLE

    try:
        if stream:
            result = read_stream(stream_url)
        elif predict:
            result = do_predict(model_filename, stream_url)
        elif generate:
            result = do_generate(count, batch_size, test_batch_size,
                                 device, epochs, log_interval, model_filename)
    except:
        print("Unexpected error:", sys.exc_info())
        result = os.EX_SOFTWARE

    return result


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
