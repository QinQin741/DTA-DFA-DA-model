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
DDCDA_evaluation_indicator = 0

def test(dataset_name):

    model_root = 'weights'
    image_root = os.path.join('F:/wangqin/Dataset', dataset_name)

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
      # model_root, 'MMD_domain_stylegan_PGGAN_model_epoch_current.pth'
        model_root, 'MMD_domain_GDWCT_FTM_model_epoch_best.pth'

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

    # s_test_accu, s_test_rec, s_test_pre, s_test_auroc = test('DFFD datasets/faceapp/test')
    # print('%s, Accuracy: %f, Recall: %f, Precision: %f, AUROC: %f' % ('source test', s_test_accu, s_test_rec, s_test_pre, s_test_auroc))

    t_test_accu, t_test_rec, t_test_pre, t_test_auroc = test('AttGAN/test')
    print('%s, Accuracy: %f, Recall: %f, Precision: %f, AUROC: %f' % ('target test', t_test_accu, t_test_rec, t_test_pre, t_test_auroc))

    # # 将指标差异的各个部分转换为 Torch 张量
    # s_test_accu = torch.tensor(s_test_accu)
    # s_test_rec = torch.tensor(s_test_rec)
    # s_test_pre = torch.tensor(s_test_pre)
    # s_test_auroc = torch.tensor(s_test_auroc)
    # t_test_accu = torch.tensor(t_test_accu)
    # t_test_rec = torch.tensor(t_test_rec)
    # t_test_pre = torch.tensor(t_test_pre)
    # t_test_auroc = torch.tensor(t_test_auroc)
    #
    # # 计算DDCDA评价指标
    # DDCDA_evaluation_indicator = torch.sqrt(torch.pow(s_test_accu - t_test_accu, 2) + torch.pow(s_test_rec - t_test_rec, 2) + torch.pow(s_test_pre - t_test_pre, 2) + torch.pow(s_test_auroc - t_test_auroc, 2)) / 4
    # print('DDCDA evaluation indicator from %s to %s： %f ' % ('stylegan', 'PGGAN', DDCDA_evaluation_indicator))




