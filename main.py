import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import numpy as np
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test_classifier import test
import torchvision





source_dataset_name = 'stylegan/train_small'
target_dataset_name = 'DFFD datasets/pggan_v1/val'
source_image_root = 'F:/wangqin/Dataset/stylegan/train'
target_image_root = 'F:/wangqin/Dataset/DFFD datasets/pggan_v1/train'
model_root = 'weights'
cuda = True
start_lr = 1e-4
batch_size = 128
image_size = 256
n_epoch = 50
source_num_images = 1000
target_num_images = 1000


manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data

img_transform_source = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.44090933, 0.30313978, 0.27302197), std=(0.21757561, 0.15543477, 0.14678976))
])

img_transform_target = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.44090933, 0.30313978, 0.27302197), std=(0.21757561, 0.15543477, 0.14678976))
])

dataset_source = torchvision.datasets.ImageFolder(
    root=source_image_root,
    transform=img_transform_source
)
source_indices = list(range(len(dataset_source)))
source_subset_indices = torch.utils.data.SubsetRandomSampler(source_indices[:source_num_images])

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=True
)

dataset_target = torchvision.datasets.ImageFolder(
    root=target_image_root,
    transform=img_transform_target
)
target_indices = list(range(len(dataset_target)))
target_subset_indices = torch.utils.data.SubsetRandomSampler(source_indices[:target_num_images])
# dataloader_target = torch.utils.data.DataLoader(
#     dataset=dataset_target,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=6,
#     drop_last=True,
#     pin_memory=True,
#     sampler=target_subset_indices
# )
dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=True
)
# load model
my_net = CNNModel()


optimizer = optim.Adam(my_net.parameters(), lr=start_lr)
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss




Resume = False
# Resume = True
if Resume:
    my_net = torch.load(os.path.join(
        model_root, 'MMD_domain_stylegan_PGGAN_model_epoch_best.pth'
    ))

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu_t = 0.0
if __name__ == "__main__":
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # adjust_learning_rate(optimizer, p, start_lr)
            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source


            batch_size = len(s_label)

            s_domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                s_domain_label = s_domain_label.cuda()



            s_class_output, s_domain_output,s_feature_output = my_net(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(s_class_output, s_label)
            err_s_domain = loss_domain(s_domain_output, s_domain_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            batch_size = len(t_img)

            t_domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                t_label = t_label.cuda()
                t_domain_label = t_domain_label.cuda()

            my_net.zero_grad()
            t_class_output, t_domain_output, t_feature_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(t_domain_output, t_domain_label)
            err_t_label = loss_class(t_class_output, t_label)
            MMD = MMDLoss()
            MMD = MMD.cuda()
            err_MMD = MMD(source = s_feature_output, target = t_feature_output)
            err = err_t_domain + err_s_domain + err_s_label+ err_MMD
            err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err: %f, err_s_label: %f, err_t_label: %f, err_s_domain: %f, err_t_domain: %f, err_MMD: %f' \
                  % (epoch, i + 1, len_dataloader, err.data.cpu().numpy(), err_s_label.data.cpu().numpy(), err_t_label.data.cpu().numpy(),
                     err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item(), err_MMD.data.cpu().numpy()))
            sys.stdout.flush()
        torch.save(my_net, '{0}/MMD_domain_stylegan_PGGAN_model_epoch_current.pth'.format(model_root))

        print('\n')

        accu_t, _, _, _ = test(target_dataset_name)
        print('Accuracy of the %s dataset: %f\n' % ('target', accu_t))
        if accu_t > best_accu_t:
            best_accu_t = accu_t
            torch.save(my_net, '{0}/MMD_domain_stylegan_PGGAN_model_epoch_best.pth'.format(model_root))

    print('============ Summary ============= \n')
    # print('Accuracy of the %s dataset: %f' % ('source', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('target', best_accu_t))
    print('Corresponding model was save in ' + model_root + '/MMD_domain_stylegan_PGGAN_model_epoch_best.pth')