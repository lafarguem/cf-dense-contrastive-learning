import torch.nn as nn
from abc import ABC, abstractmethod
from pretraining.models.contrastive.output import SingleBranchContrastiveOutput, DualBranchContrastiveOutput
import torch
import torch.nn as nn

class SingleBranchContrastiveModel(nn.Module, ABC):
    def __init__(self, dense_projection_head=[], instance_projection_head=[], disable_instance=False, has_decoder=False, return_global=False, return_dense=True, runtime_args=None):
        super().__init__()
        self.dense_projection_head = dense_projection_head
        self.instance_projection_head = instance_projection_head
        self.disable_instance = disable_instance
        self.has_decoder=has_decoder
        self.return_global = return_global
        self.return_dense = return_dense
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_channels = None
        self.encoder_out_channels = None
    
    def setup_head(self):
        if isinstance(self.dense_projection_head, nn.Module):
            return
        if self.dense_projection_head == []:
            self.dense_projection_head = nn.Identity()
            return
        channels = [self.out_channels] + self.dense_projection_head
        head_list = []
        for i in range(0, len(channels) - 1):
            head_list.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=1))
            if i != len(channels) - 2:
                head_list.append(nn.ReLU(inplace=True))
        self.dense_projection_head = nn.Sequential(*head_list)

        if self.disable_instance:
            return
        if self.instance_projection_head == []:
            self.instance_projection_head = nn.Identity()
            return
        channels = [self.encoder_out_channels] + self.instance_projection_head
        head_list = []
        for i in range(0, len(channels) - 1):
            head_list.append(nn.Linear(channels[i], channels[i+1]))
            if i != len(channels) - 2:
                head_list.append(nn.ReLU(inplace=True))
        self.instance_projection_head = nn.Sequential(*head_list)

    def set_toggle(self, toggle):
        if toggle == 'both':
            self.set_dense_return()
            self.set_global_return()
        elif toggle == 'dense':
            self.set_dense_return()
            self.set_global_return(False)
        elif toggle == 'global':
            self.set_dense_return(False)
            self.set_global_return()
        else:
            raise ValueError("Please pass toggle with value 'both', 'dense', or 'global'")

    def set_global_return(self, value=True):
        self.return_global = value
    
    def set_dense_return(self, value=True):
        self.return_dense = value
    
    @abstractmethod
    def encode(self, input):
        pass

    @abstractmethod
    def decode(self, embeddings):
        pass

    @abstractmethod
    def get_pretrained_model(self):
        pass

    def encoded_for_global(self,encoding):
        return encoding
    
    def encoded_for_dense(self,encoding):
        return encoding

    def forward(self, input):
        coords = input.get('coords', None)
        instance_embeddings = None
        projected_instance_embeddings = None
        encoding = self.encode(input)
        if self.return_global:
            embeddings = self.encoded_for_global(encoding)
            instance_embeddings = self.avg_pool(embeddings)
            instance_embeddings = instance_embeddings.flatten(start_dim=2)
            B,V,C = instance_embeddings.shape
            projected_instance_embeddings = self.instance_projection_head(instance_embeddings.view(B*V, C))
            projected_instance_embeddings = projected_instance_embeddings.view(B,V,-1)
        if self.has_decoder and self.return_dense:
            embeddings = self.decode(encoding)
            B,V,C,H,W  = embeddings.shape
            projected_embeddings = self.dense_projection_head(embeddings.view(B*V,C,H,W)).view(B,V,-1,H,W)
        elif not self.has_decoder:
            embeddings = self.encoded_for_dense(encoding)
            B,V,C,H,W  = embeddings.shape
            projected_embeddings = self.dense_projection_head(embeddings.view(B*V,C,H,W)).view(B,V,-1,H,W)
        else:
            embeddings = None
            projected_embeddings = None
        out = SingleBranchContrastiveOutput(dense_embeddings=embeddings, 
                                            instance_embeddings=instance_embeddings, 
                                            projected_dense_embeddings=projected_embeddings,
                                            projected_instance_embeddings=projected_instance_embeddings,
                                            coords=coords)
        return out
    
    def evaluate(self, input):
        encoding = self.encode(input)
        if self.has_decoder:
            return self.decode(encoding)
        else:
            return self.encoded_for_global(encoding)

class DualBranchContrastiveModel(nn.Module, ABC):
    def __init__(self, student: SingleBranchContrastiveModel, teacher: SingleBranchContrastiveModel, 
                 return_global=False, return_dense=True, runtime_args=None):
        super().__init__()
        self.student = student
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.return_global = return_global
        self.return_dense = return_dense

    def set_toggle(self, toggle):
        self.student.set_toggle(toggle)
        self.teacher.set_toggle(toggle)

    def set_global_return(self, value=True):
        self.student.set_global_return(value)
        self.teacher.set_global_return(value)
    
    def set_dense_return(self, value=True):
        self.student.set_dense_return(value)
        self.teacher.set_dense_return(value)
    
    def get_pretrained_model(self):
        return self.student.get_pretrained_model()

    @abstractmethod
    def update_teacher(self):
        pass

    def forward(self, input):
        student_out = self.student(input)
        with torch.no_grad():
            self.update_teacher()
            teacher_out = self.teacher(input)
        out = DualBranchContrastiveOutput(
            dense_embeddings=student_out.dense_embeddings,
            teacher_dense_embeddings=teacher_out.dense_embeddings,
            instance_embeddings=student_out.instance_embeddings,
            teacher_instance_embeddings=teacher_out.instance_embeddings,
            projected_dense_embeddings=student_out.projected_dense_embeddings,
            projected_teacher_dense_embeddings=teacher_out.projected_dense_embeddings,
            projected_instance_embeddings=student_out.projected_instance_embeddings,
            projected_teacher_instance_embeddings=teacher_out.projected_instance_embeddings,
            coords=student_out.coords
        )
        return out
    
    def evaluate(self, input):
        return self.student.evaluate(input)