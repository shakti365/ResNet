import torch
import torchvision
import torchvision.transforms as transforms
from absl import app
from absl import flags
from absl import logging
from sklearn.metrics import classification_report

from resnet import Net

torch.manual_seed(0)

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", "../data", "Path to store dataset")
flags.DEFINE_boolean("debug", False, "Runs in debug mode")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_string("model_path", "../model/model_2.pth", "model path")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def download_test_data(data_path, transform=None):
    """
    Downloads the CIFAR10 dataset to data_path.
    Doesn't download if it already exists.
    """
    # Get the CIFAR 10 data


    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           transform=transform, download=True)

    logging.debug(testset)


    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=False, num_workers=1)

    return testloader


def get_transform():
    """
    conver into torch tensor
    """
    transform = transforms.Compose([
        transforms.ToTensor()
        ])

    return transform

def accuracy(true,pred):
    acc = (true == pred.argmax(-1)).float().detach().cpu().numpy()
    return float(100 * acc.sum() / float(len(acc)))

def main(argv):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)

    # Transform and Load the dataset
    transform = get_transform()
    testloader = download_test_data(FLAGS.data_path, transform)

    # Get the model
    model = Net()
    checkpoint = torch.load(FLAGS.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    if torch.cuda.is_available():
        model = model.to(device)

    total_acc = []
    all_targets = []
    all_pred = []
    for i, data in enumerate(testloader):

        # Take the inputs and labels
        inputs, labels = data

        all_targets.extend(labels.detach().numpy())

        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        all_pred.extend(outputs.argmax(-1).detach().cpu().numpy())

        acc_batch = accuracy(labels, outputs)
        total_acc.append(acc_batch)
        logging.info(f"Batch Accuracy: {acc_batch}")

    avg_acc = sum(total_acc) / float(len(total_acc))
    logging.info(f"Average Accuracy: {avg_acc}")
    logging.info(classification_report(all_targets, all_pred,target_names=classes))


if __name__ == "__main__":
    app.run(main)
