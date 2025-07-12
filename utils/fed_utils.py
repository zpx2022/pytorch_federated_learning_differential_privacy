from utils.models import *

def assign_dataset(dataset_name):
    """
    Assigns dataset-specific parameters based on its name.
    :param dataset_name: The name of the dataset.
    :return: A tuple containing (num_class, image_dim, image_channel).
    """
    if dataset_name == 'MNIST':
        return 10, 28, 1
    elif dataset_name == 'FashionMNIST':
        return 10, 28, 1
    elif dataset_name == 'EMNIST':
        return 27, 28, 1
    elif dataset_name == 'CIFAR10':
        return 10, 32, 3
    elif dataset_name == 'CIFAR100':
        return 100, 32, 3
    elif dataset_name == 'SVHN':
        return 10, 32, 3
    elif dataset_name == 'IMAGENET':
        return 200, 64, 3
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def init_model(model_name, num_class, image_channel):
    """
    Initializes a model for a specific learning task.
    :param model_name: The name of the model architecture.
    :param num_class: The number of classes in the dataset.
    :param image_channel: The number of image channels.
    :return: The initialized model instance.
    """
    model = None
    if model_name.startswith("ResNet"):
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name.startswith("VGG"):
        model = generate_vgg(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "LeNet":
        model = LeNet(num_classes=num_class, in_channels=image_channel)
    elif model_name == "CNN":
        model = CNN(num_classes=num_class, in_channels=image_channel)
    elif model_name == "AlexCifarNet":
        model = AlexCifarNet()
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model
