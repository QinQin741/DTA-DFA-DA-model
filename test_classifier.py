import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
import torchvision
from model import CNNModel
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch


def test(dataset_name):

    model_root = 'weights'
    image_root = dataset_name

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 256
    alpha = 0
    # 使用训练好的模型进行预测
    predictions = []
    true_labels = []
    scores = []

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.44078606, 0.30346632, 0.27303344), std=(0.21736373, 0.1552905, 0.14665934))
    ])

    dataset = torchvision.datasets.ImageFolder(
        root=image_root,
        transform=img_transform_source
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )



    """ test """
    my_net = torch.load(os.path.join(
         model_root, 'current.pth'
       #  model_root, 'best.pth'

    ))



    if cuda:
        my_net = my_net.cuda()

    for inputs, labels in dataloader:
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        class_output, _, _ = my_net(inputs, alpha=0)
        _, predicted = torch.max(class_output.data, 1)

        scores.extend(class_output[:, 1].tolist())
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())
    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    fpr, tpr, _ = roc_curve(true_labels, scores)
    auroc = auc(fpr, tpr)
    return accuracy, recall, precision, auroc

if __name__ == "__main__":

   
    t_test_accu, t_test_rec, t_test_pre, t_test_auroc = test('AttGAN/test')
    print('%s, Accuracy: %f, Recall: %f, Precision: %f, AUROC: %f' % ('target test', t_test_accu, t_test_rec, t_test_pre, t_test_auroc))


