import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Subset
import os
import PIL

def load_data(name, root='./data', download=True):
    """
    Loads a specified dataset from torchvision.
    :param name: The name of the dataset (e.g., 'MNIST', 'CIFAR10').
    :param root: The root directory for the dataset.
    :param download: Whether to download the dataset if not found.
    :return: A tuple of (trainset, testset, num_classes).
    """
    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN', "IMAGENET", 'CIFAR100']
    if name not in data_dict:
        raise ValueError(f"Dataset {name} not supported.")

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    # Define transformations based on dataset
    if name in ['MNIST', 'EMNIST', 'QMNIST']:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif name == 'CIFAR100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif name == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Add other transformations as needed

    # Load datasets
    if name == 'MNIST':
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
    elif name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    # Add other dataset loading logic here
    
    # Ensure targets are tensors
    if hasattr(trainset, 'targets') and not isinstance(trainset.targets, torch.Tensor):
        trainset.targets = torch.tensor(trainset.targets)
    if hasattr(testset, 'targets') and not isinstance(testset.targets, torch.Tensor):
        testset.targets = torch.tensor(testset.targets)

    num_classes = len(trainset.classes) if hasattr(trainset, 'classes') else trainset.targets.max().item() + 1
    
    return trainset, testset, num_classes

def divide_data(num_client=1, num_local_class=10, dataset_name='emnist', i_seed=0):
    """
    Divides the training dataset among clients to simulate a non-IID distribution.
    Each client receives data from a specific number of classes.
    :param num_client: The total number of clients.
    :param num_local_class: The number of distinct classes on each client.
    :param dataset_name: The name of the dataset to use.
    :param i_seed: The random seed for reproducibility.
    :return: A tuple of (trainset_config, testset).
    """
    torch.manual_seed(i_seed)

    trainset, testset, num_classes = load_data(dataset_name, download=True)

    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "Number of local classes must be between 1 and total classes."

    # --- Step 1: Determine class distribution for each client ---
    config_class = {f'f_{i:05d}': [] for i in range(num_client)}
    config_division = {cls: 0 for cls in range(num_classes)}
    for i in range(num_client):
        client_name = f'f_{i:05d}'
        for j in range(num_local_class):
            cls = (i + j) % num_classes
            config_class[client_name].append(cls)
            config_division[cls] += 1
            
    # --- Step 2: Partition data indices by class ---
    config_data = {cls: [] for cls in range(num_classes)}
    class_indices = {cls: (trainset.targets == cls).nonzero().squeeze() for cls in range(num_classes)}
    
    for cls in range(num_classes):
        # Shuffle indices for randomness
        perm = torch.randperm(len(class_indices[cls]))
        shuffled_indices = class_indices[cls][perm]
        
        # Partition the shuffled indices for clients that need this class
        num_partitions = config_division[cls]
        partitions = torch.tensor_split(shuffled_indices, num_partitions)
        config_data[cls] = list(partitions)

    # --- Step 3: Assign data partitions to clients ---
    trainset_config = {'users': [], 'user_data': {}, 'num_samples': []}
    # Keep track of which partition to assign next for each class
    class_partition_pointers = {cls: 0 for cls in range(num_classes)}

    for user in tqdm(sorted(config_class.keys()), desc="Dividing data"):
        user_data_indices = []
        for cls in config_class[user]:
            partition_idx = class_partition_pointers[cls]
            user_data_indices.append(config_data[cls][partition_idx])
            class_partition_pointers[cls] += 1
        
        # Combine all indices for the current user
        user_data_indices = torch.cat(user_data_indices).tolist()
        
        # Create a subset for the user
        user_subset = Subset(trainset, user_data_indices)
        
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_subset
        trainset_config['num_samples'].append(len(user_subset))

    return trainset_config, testset
