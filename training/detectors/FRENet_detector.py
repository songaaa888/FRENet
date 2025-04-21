import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)


class SoftPooling(nn.Module):
    def __init__(self, kernel_size=7, stride=3, padding=0):
        super(SoftPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool

class PatchFocus(nn.Module):
    """ 定义局部注意力机制 """
    def __init__(self, channels, f=16, local_kernel=7, stride=3):
        super().__init__()
        self.local_kernel = local_kernel  # 局部感知窗口大小
        self.stride = stride  # 步长决定局部区域覆盖范围

        self.body = nn.Sequential(
            nn.Conv2d(channels, f, 1),  # 局部特征提取
            SoftPooling(kernel_size=self.local_kernel, stride=self.stride),  # 软池化
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),  # 进一步提取局部模式
            nn.Conv2d(f, channels, 3, padding=1),  # 通道恢复
            nn.Sigmoid(),  # 生成局部注意力热图
        )

        self.gate = nn.Sequential(
            nn.Sigmoid(),  # 额外的特征门控机制
        )

    def forward(self, x):
        ''' Forward 计算局部权重 '''
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        g = self.gate(x[:, :1].clone())  # 仅用第一个通道计算特征门控

        return x * w * g  # 应用局部注意力

@DETECTOR.register_module(module_name='FRENet')
class FRENet(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.RefFus = RefFus(in_channels1=2048, k_dim=64, v_dim=64, num_heads=8)  
        self.local_attention = PatchFocus(2048)
    
    def build_backbone(self, config):
        # 获取 backbone（如 Xception）
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        # 加载预训练模型
        state_dict = torch.load(config['pretrained'])

        # **调整 pointwise 权重维度**
        for name, weights in state_dict.items():
            if 'pointwise' in name and len(weights.shape) == 2:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)  

  
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

  
        conv1_data = state_dict.pop('conv1.weight')

        backbone.load_state_dict(state_dict, strict=False)  # 允许部分不匹配
        logger.info('Load pretrained model from {}'.format(config['pretrained']))

        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)
        backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)
        logger.info('Copy conv1 from pretrained model')

        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict, phase_fea) -> tuple:

        features = torch.cat((data_dict['image'], phase_fea), dim=1)
        
        # 通过 Backbone 提取特征
        feat = self.backbone.features(features)  # 身份特征
        
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        loss_id = self.loss_func(pred_dict['cls_id'], label)
        
        loss_dict = {'overall': loss_id}  # 计算损失
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        auc_id, eer_id, acc_id, ap_id = calculate_metrics_for_train(label.detach(), pred_dict['prob_id'].detach())

        metric_batch_dict = {
            'acc_id': acc_id, 'auc_id': auc_id, 'eer_id': eer_id, 'ap_id': ap_id,
        }
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        phase_fea = self.phase_without_amplitude(data_dict['image'])
        feat = self.features(data_dict, phase_fea)

        feat_noise = torch.randn_like(feat)

        pure_feat = self.RefFus(feat, feat_noise)

        id_pred = self.classifier(pure_feat)

        # 确保 `prob_id` 维度正确
        prob_id = torch.softmax(id_pred, dim=1)  # (batch_size, 2)

        pred_dict = {
            'cls_id': id_pred, 'prob_id': prob_id, 'feat_id': pure_feat,
            'prob': prob_id[:, 1], 
        }

        return pred_dict

    def phase_without_amplitude(self, img):
        # Convert to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True) # shape: (batch_size, 1, 256, 256)
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img,dim=(-1,-2))
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X,dim=(-1,-2)))
        return reconstructed_x

class LPSA(nn.Module):
    def __init__(self, in_channels1, k_dim, v_dim, num_heads):
        super(LPSA, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        # 低秩投影 (Low-Rank Projection)
        self.low_rank_proj_q = nn.Linear(in_channels1, k_dim // 2, bias=False)
        self.low_rank_proj_k = nn.Linear(in_channels1, k_dim // 2, bias=False)
        self.low_rank_proj_v = nn.Linear(in_channels1, v_dim // 2, bias=False)

        # 交叉注意力 (Cross Attention)
        self.proj_q = nn.Linear(k_dim // 2, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(k_dim // 2, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(v_dim // 2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_channels1)


        self.local_attn = nn.Conv2d(in_channels1, in_channels1, kernel_size=3, stride=1, padding=1, groups=in_channels1)
        

        self.feature_gate = nn.Sequential(
            nn.Conv2d(in_channels1, in_channels1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        batch_size, channels1, height1, width1 = x.size()

        # 低秩投影降维
        q = self.low_rank_proj_q(x.permute(0, 2, 3, 1).contiguous().view(batch_size, height1 * width1, channels1))
        k = self.low_rank_proj_k(x.permute(0, 2, 3, 1).contiguous().view(batch_size, height1 * width1, channels1))
        v = self.low_rank_proj_v(x.permute(0, 2, 3, 1).contiguous().view(batch_size, height1 * width1, channels1))

        # 计算注意力
        q = self.proj_q(q).view(batch_size, height1 * width1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k = self.proj_k(k).view(batch_size, height1 * width1, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v = self.proj_v(v).view(batch_size, height1 * width1, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) / self.k_dim**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, height1 * width1, -1)
        output = self.proj_o(output)
        
        output = output.view(batch_size, height1, width1, channels1).permute(0, 3, 1, 2).contiguous()
        output = self.local_attn(output)
        gate = self.feature_gate(output)
        output = output * gate

        return output

class RefFus(nn.Module):
    def __init__(self, in_channels1, k_dim, v_dim, num_heads):
        super(RefFus, self).__init__()
        self.feat_crossattn = LPSA(in_channels1, k_dim, v_dim, num_heads)

        self.local_attention = PatchFocus(in_channels1)  
        
        self.feat_residual = nn.Conv2d(in_channels1, in_channels1, 1, bias=False)

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels1, in_channels1, 1, bias=True),
            nn.Sigmoid()
        )

        self.conv256 = nn.Conv2d(in_channels1, in_channels1, 1, bias=True)

    def forward(self, feat, feat_noise, mask=None):
        feat_weight = self.feat_crossattn(feat)

        feat_weight = self.feat_residual(feat_weight) + feat

        mix_weight = self.local_attention(feat_weight)

        pure_feat = (1 - mix_weight) * feat + mix_weight * feat_noise
        pure_feat = self.conv256(pure_feat)

        return pure_feat


