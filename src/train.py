import torch
import torchvision
import torchvision.transforms as transforms
from absl import app
from absl import flags
from absl import logging
import os
from resnet import Net

torch.manual_seed(0)

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", "../data", "Path to store dataset")
flags.DEFINE_boolean("debug", False, "Runs in debug mode")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("epochs", 3, "Number of epochs to run the training")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_string("save_model_dir", "../model", "Path to dir to save model")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def download_data(data_path, transform=None):
    """
    Downloads the CIFAR10 dataset to data_path.
    Doesn't download if it already exists.
    """
    # Get the CIFAR 10 data
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            transform=transform, download=True)

    trainset, valset = torch.utils.data.random_split(dataset, [45000, 5000])

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           transform=transform, download=True)


    logging.debug(trainset)
    logging.debug(valset)
    logging.debug(testset)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=FLAGS.batch_size,
                                              shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(valset,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=False, num_workers=1)

    return trainloader, valloader


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

    if not os.path.exists(FLAGS.save_model_dir):
        os.makedirs(FLAGS.save_model_dir)

    # Transform and Load the dataset
    transform = get_transform()
    trainloader, valloader = download_data(FLAGS.data_path, transform)

    # Get the model
    model = Net()
    if torch.cuda.is_available():
        model = model.to(device)

    # Create the loss fn and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    for epoch in range(FLAGS.epochs):

        epoch_loss = []
        for i, data in enumerate(trainloader):

            # Take the inputs and labels
            inputs, labels = data

            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
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

        # get val loss
        val_epoch_loss = []
        last_val_loss = 100
        for i, data in enumerate(valloader):

            # Take the inputs and labels
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)

            # Do a backprop
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            # Log statistics
            val_epoch_loss.append(loss.item())


        val_avg_loss = sum(val_epoch_loss) / len(val_epoch_loss)
        logging.info(f"Epoch: {epoch}\tAverage Val Loss: {val_avg_loss}")

        if val_avg_loss<last_val_loss:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_avg_loss},  os.path.join(FLAGS.save_model_dir, "model_{}.pth".format(epoch)))
            last_val_loss = val_avg_loss


    "------------------------FINISHED TRAINING--------------------------"


if __name__ == "__main__":
    app.run(main)
