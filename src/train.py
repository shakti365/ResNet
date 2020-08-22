import torch
import torchvision
import torchvision.transforms as transforms
from absl import app
from absl import flags
from absl import logging

from resnet import Net


FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", "../data", "Path to store dataset")
flags.DEFINE_boolean("debug", False, "Runs in debug mode")
flags.DEFINE_integer("batch_size", 4, "Batch size")
flags.DEFINE_integer("epochs", 2, "Number of epochs to run the training")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")


def download_data(data_path, transform=None):
    """
    Downloads the CIFAR10 dataset to data_path.
    Doesn't download if it already exists.
    """
    # Get the CIFAR 10 data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            transform=transform, download=True)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           transform=transform, download=True)

    logging.debug(trainset)
    logging.debug(testset)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=FLAGS.batch_size,
                                              shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=True, num_workers=1)

    return trainloader, testloader


def get_transform():
    # Perform data augmentation
    # Pad with 4 pixels on all the sides
    # Randomly flip the image along horizontal axis
    # Crop 32X32 image from the padded image
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32),
        transforms.ToTensor()
        ])

    return transform


def main(argv):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)

    # Transform and Load the dataset
    transform = get_transform()
    trainloader, testloader = download_data(FLAGS.data_path, transform)

    # Get the model
    model = Net()

    # Create the loss fn and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    for epoch in range(FLAGS.epochs):

        epoch_loss = []
        for i, data in enumerate(trainloader):

            # Take the inputs and labels
            inputs, labels = data

            # Make sure the grads are zero before a step
            optimizer.zero_grad()

            # Do a backprop
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log statistics
            epoch_loss.append(loss.item())
            if i % 1000 == 0:
                logging.info(f"Epoch: {epoch}\tIteration: {i}\tLoss: {loss}")

        avg_loss = sum(epoch_loss) / len(epoch_loss) 
        logging.info(f"Epoch: {epoch}\tAverage Loss: {avg_loss}")


if __name__ == "__main__":
    app.run(main)
