from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
from functions import ReverseLayerF




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Feature extraction block
        self.resnet_model = models.resnet18(weights="IMAGENET1K_V1")
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.layer3 = self.resnet_model.layer3
        self.avgpool = self.resnet_model.avgpool
    
        # Classification block
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(256, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))
        
        # Tag adversarial block
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        
        # Feature alignment block
        self.MMD_classifier = nn.Sequential()
        self.MMD_classifier.add_module('m_fc1', nn.Linear(256, 256))
        self.MMD_classifier.add_module('m_bn1', nn.BatchNorm1d(256))
        self.MMD_classifier.add_module('m_relu1', nn.ReLU(True))
        self.MMD_classifier.add_module('m_fc2', nn.Linear(256, 256))
        self.MMD_classifier.add_module('m_bn2', nn.BatchNorm1d(256))
        self.MMD_classifier.add_module('m_relu2', nn.ReLU(True))
        self.MMD_classifier.add_module('m_fc3', nn.Linear(256, 2))
        self.MMD_classifier.add_module('m_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        class_output = self.class_classifier(feature)
        feature_output = self.MMD_classifier(feature)

        return class_output, domain_output, feature_output
