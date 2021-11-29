import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from imageio import imread
import torchvision

class PerceptLoss(object):

    def __init__(self):
        pass

    def __call__(self, LossNet, fake_img, real_img):
        with torch.no_grad():
            real_feature = LossNet(real_img.detach())
        fake_feature = LossNet(fake_img)
        perceptual_penalty = F.mse_loss(fake_feature, real_feature)
        return perceptual_penalty

    def set_ftr_num(self, ftr_num):
        pass


class DiscriminatorLoss(object):

    def __init__(self, ftr_num=4, data_parallel=False):
        self.data_parallel = data_parallel
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img):
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_img.detach())
            d, fake_feature = nn.parallel.data_parallel(D, fake_img)
        else:
            with torch.no_grad():
                d, real_feature = D(real_img.detach())
            d, fake_feature = D(fake_img)
        D_penalty = 0
        for i in range(self.ftr_num):
            f_id = -i - 1
            D_penalty = D_penalty + F.l1_loss(fake_feature[f_id],
                                              real_feature[f_id])
        return D_penalty

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num

class VGGPerceptualLoss(torch.nn.Module):

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_, target, feature_layers=[0, 1, 2, 3], style_layers=[3]):
        if input_.shape[1] != 3:
            input_ = input_.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # print(type(self.mean))
        # print(type(input_))
        # print(type(self.std))
        # input()
        # print(input_)
        # print(target)
        input_ =  torch.div( torch.sub(input_,self.mean) , self.std)
        target =  torch.div( torch.sub(target,self.mean) , self.std)
        if self.resize:
            input_ = self.transform(input_, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input_
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
        
########################################################### Test Code ########################################################
# input_ = imread('/home/haneollee/dgm3/edge-connect/checkpoints/results/dgm_final/1_edges_sobel/images_0_merged.png')
# target = imread('/home/haneollee/dgm3/edge-connect/checkpoints/results/dgm_final/1_edges_sobel/images_0_unmerged.png')

# vgg_perceptual_loss = VGGPerceptualLoss(resize=True)
# loss_result = vgg_perceptual_loss.forward(torch.from_numpy(input_/255.).float(), torch.from_numpy(target/255.).float(), feature_layers=[0, 1, 2, 3] , style_layers=[2,3] )
# print('# ex1> Merged_0 and Unmerged_0')
# print(f'loss_result = {loss_result}\n')


# input_ = imread('/home/haneollee/dgm3/edge-connect/checkpoints/results/dgm_final/2_edges_merged/images_0_edges_merged.png')
# target = imread('/home/haneollee/dgm3/edge-connect/checkpoints/results/dgm_final/edges_middle/images_0_edges_middle_masked.png')

# loss_result = vgg_perceptual_loss.forward(torch.from_numpy(input_/255.).float(), torch.from_numpy(target/255.).float(), feature_layers=[0, 1, 2, 3] , style_layers=[2,3] )
# print('# ex2> images_0_edges_merged.png and images_0_edges_middle_masked.png')
# print(f'loss_result = {loss_result}\n')

# input_, target are numpy.array datatype
# input_ = imread('/home/haneollee/dgm3/edge-connect/checkpoints/results/dgm_final/2_edges_merged/images_0_edges_merged.png')
# target = imread('/home/haneollee/dgm3/edge-connect/checkpoints/results/dgm_final/2_edges_merged/images_0_edges_merged.png')

# loss_result = vgg_perceptual_loss.forward(torch.from_numpy(input_/255.).float(), torch.from_numpy(target/255.).float(), feature_layers=[0, 1, 2, 3] , style_layers=[2,3] )
# print('# ex3> same images')
# print(f'loss_result = {loss_result}\n')
