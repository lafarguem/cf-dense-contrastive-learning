# Code adapted from https://github.com/zdaxie/PixPro
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pretraining.utils.distributed import get_world_size
from pretraining.utils.common import getattr_nested
from pretraining.models.contrastive.base_model import SingleBranchContrastiveModel, DualBranchContrastiveModel

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x

def Proj_Head(in_dim=2048, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)
    
class PixProStudent(SingleBranchContrastiveModel):
    def __init__(
            self,
            pixpro_p, 
            pixpro_clamp_value,
            pixpro_transform_layer,
            backbone = None,
            return_global=False, 
            return_dense=True,
            disable_instance=False,
        ):
        has_decoder = backbone.has_decoder
        super().__init__(has_decoder, return_global, return_dense)
        self.backbone = backbone
        self.pixpro_p               = pixpro_p
        self.pixpro_clamp_value     = pixpro_clamp_value
        self.pixpro_transform_layer = pixpro_transform_layer

        self.projector = Proj_Head(self.backbone.out_channels)

        if self.pixpro_transform_layer == 0:
            self.value_transform = Identity()
        elif self.pixpro_transform_layer == 1:
            self.value_transform = conv1x1(in_planes=256, out_planes=256)
        elif self.pixpro_transform_layer == 2:
            self.value_transform = MLP2d(in_dim=256, inner_dim=256, out_dim=256)
        else:
            raise NotImplementedError

        if not disable_instance:
            self.projector_instance = Proj_Head()
            self.predictor = Pred_Head()

        if get_world_size() > 1:
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
            self.projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
            if not disable_instance:
                self.projector_instance = nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
                self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

    def featprop(self, feat):
        N, C, H, W = feat.shape

        feat_value = self.value_transform(feat)
        feat_value = F.normalize(feat_value, dim=1, eps=1e-8)
        feat_value = feat_value.view(N, C, -1)

        feat = F.normalize(feat, dim=1, eps=1e-8)

        feat = feat.view(N, C, -1)

        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.:
            attention = attention + 1e-6
        attention = attention ** self.pixpro_p

        feat = torch.bmm(feat_value, attention.transpose(1, 2))

        return feat.view(N, C, H, W)
    
    def encode(self, input):
        return self.backbone.encode(input)
    
    def decode(self, encoding):
        B,V,C,H,W = encoding[1].shape
        features = self.backbone.decode(encoding)
        proj = self.projector(features.view(B*V,C,H,W))
        pred = self.featprop(proj)
        pred = F.normalize(pred, dim=-3, eps=1e-8)
        return pred

    def encoded_for_global(self, encoding):
        _, deepest = encoding
        B,V,C,H,W = deepest.shape
        proj_instance = self.projector_instance(deepest.view(B*V,C,H,W))
        pred_instance = self.predictor(proj_instance)
        return pred_instance.view(B,V,*pred_instance.shape[1:])
    
    def encoded_for_dense(self, encoding):
        _, deepest = encoding
        B,V,C,H,W = deepest.shape
        proj = self.projector(deepest.view(B*V,C,H,W))
        pred = self.featprop(proj)
        pred = F.normalize(pred, dim=-3, eps=1e-8)
        return pred.view(B,V,*pred.shape[1:])

    def evaluate(self, input):
        encoding = self.backbone.encode(input)
        if self.has_decoder:
            B,V,C,H,W = encoding[1].shape
            features = self.backbone.decode(encoding)
            return features
        else:
            return encoding[1]

class PixProTeacher(SingleBranchContrastiveModel):
    def __init__(
            self,
            backbone = None,
            return_global=False, 
            return_dense=True,
            disable_instance=False,
        ):
        has_decoder = backbone.has_decoder
        super().__init__(has_decoder, return_global, return_dense)

        self.backbone = backbone

        self.projector = Proj_Head(self.backbone.out_channels)

        if get_world_size() > 1:
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
            self.projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)

        if not disable_instance:
            self.projector_instance = Proj_Head()

        if get_world_size() > 1:
            if not disable_instance:
                self.projector_instance = nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)

    def encode(self, input):
        return self.backbone.encode(input)
    
    def decode(self, encoding):
        B,V,C,H,W = encoding[1].shape
        features = self.backbone.decode(encoding)
        proj = self.projector(features.view(B*V,C,H,W))
        proj = F.normalize(proj, dim=-3, eps=1e-8)
        return proj

    def encoded_for_global(self, encoding):
        _, deepest = encoding
        B,V,C,H,W = deepest.shape
        proj_instance = self.projector_instance(deepest.view(B*V,C,H,W))
        return proj_instance.view(B,V,*proj_instance.shape[1:])
    
    def encoded_for_dense(self, encoding):
        _, deepest = encoding
        B,V,C,H,W = deepest.shape
        proj = self.projector(deepest.view(B*V,C,H,W))
        proj = F.normalize(proj, dim=-3, eps=1e-8)
        return proj.view(B,V,*proj.shape[1:])


class DualBranchPixPro(DualBranchContrastiveModel):
    def __init__(self, student, teacher, momentum, return_global=False, return_dense=True, runtime_args=None,
                 always_update = ['backbone', 'projector'], global_update = ['projector_instance']):
        super().__init__(student, teacher, return_global, return_dense)
        self.momentum = momentum
        self.always_update = always_update
        self.global_update = global_update
        self.K = int(runtime_args.num_instances * 1. / get_world_size() / runtime_args.batch_size * runtime_args.epochs)
        self.k = int(runtime_args.num_instances * 1. / get_world_size() / runtime_args.batch_size * (runtime_args.start_epoch - 1))

        if self.always_update == []:
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            return
            
        for name in self.always_update:
            student_module = getattr_nested(self.student, name)
            teacher_module = getattr_nested(self.teacher, name)
            for param_q, param_k in zip(student_module.parameters(), teacher_module.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        for name in self.global_update:
            student_module = getattr_nested(self.student, name)
            teacher_module = getattr_nested(self.teacher, name)
            for param_q, param_k in zip(student_module.parameters(), teacher_module.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
    
    @torch.no_grad()
    def update_teacher(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        if self.always_update == []:
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
            return
            
        for name in self.always_update:
            student_module = getattr_nested(self.student, name)
            teacher_module = getattr_nested(self.teacher, name)
            for param_q, param_k in zip(student_module.parameters(), teacher_module.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        if self.return_global:
            for name in self.global_update:
                student_module = getattr_nested(self.student, name)
                teacher_module = getattr_nested(self.teacher, name)
                for param_q, param_k in zip(student_module.parameters(), teacher_module.parameters()):
                    param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)