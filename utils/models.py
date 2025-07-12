import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

class AlexCifarNet(nn.Module):
    """AlexNet model customized for CIFAR-10."""
    def __init__(self):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class LeNet(nn.Module):
    """LeNet model customized for MNIST-like datasets."""
    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out

def generate_resnet(num_classes=10, in_channels=1, model_name="ResNet18"):
    """Generates a ResNet model from torchvision and adapts it."""
    resnet_map = {
        "ResNet18": models.resnet18,
        "ResNet34": models.resnet34,
        "ResNet50": models.resnet50,
        "ResNet101": models.resnet101,
        "ResNet152": models.resnet152,
    }
    model = resnet_map[model_name](pretrained=True)
    # Adapt the first convolutional layer for different input channels
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Adapt the final fully connected layer for the number of classes
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def generate_vgg(num_classes=10, in_channels=1, model_name="vgg11"):
    """Generates a VGG model from torchvision and adapts it."""
    vgg_map = {
        "VGG11": models.vgg11,
        "VGG11_bn": models.vgg11_bn,
        "VGG13": models.vgg13,
        "VGG13_bn": models.vgg13_bn,
        "VGG16": models.vgg16,
        "VGG16_bn": models.vgg16_bn,
        "VGG19": models.vgg19,
        "VGG19_bn": models.vgg19_bn,
    }
    model = vgg_map[model_name](pretrained=True)
    # Adapt the final classifier layer for the number of classes
    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)
    return model

class CNN(nn.Module):
    """A custom CNN architecture."""
    def __init__(self, num_classes=10, in_channels=1):
        super(CNN, self).__init__()
        self.fp_con1 = nn.Sequential(OrderedDict([
            ('con0', nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.ternary_con2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        self.fp_fc = nn.Linear(4096, num_classes, bias=False)

    def forward(self, x):
        x = self.fp_con1(x)
        x = self.ternary_con2(x)
        x = x.view(x.size(0), -1)
        x = self.fp_fc(x)
        output = F.log_softmax(x, dim=1)
        return output
