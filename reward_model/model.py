import torch
import torch.nn as nn
from torchvision import models

class SwinTransformer(nn.Module):
    def __init__(self, model_name='swin_v2_t'):
        super(SwinTransformer, self).__init__()
        
        if model_name == 'swin_v2_t':
            self.base_model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        elif model_name == 'swin_v2_s':
            self.base_model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
        elif model_name == 'swin_v2_b':
            self.base_model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Invalid model name. Choose from 'swin_v2_t', 'swin_v2_s', 'swin_v2_b'.")
        
        # Modify the input layer to accept 1-channel input
        self.base_model.features[0][0] = nn.Conv2d(1, self.base_model.features[0][0].out_channels,
                                                   kernel_size=self.base_model.features[0][0].kernel_size,
                                                   stride=self.base_model.features[0][0].stride,
                                                   padding=self.base_model.features[0][0].padding,
                                                   bias=False)
        
        # Replace the classifier layer with an identity layer
        self.fc = nn.Linear(self.base_model.head.in_features, 1)
        self.base_model.head = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super(EfficientNet, self).__init__()
        if model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b1':
            self.base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b2':
            self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b3':
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b4':
            self.base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b5':
            self.base_model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b6':
            self.base_model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b7':
            self.base_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_v2_s':
            self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_v2_m':
            self.base_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_v2_l':
            self.base_model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Invalid model name. Choose from 'efficientnet_b0' to 'efficientnet_b7' and 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'.")
        
        # Modify the input layer to accept 1-channel input
        self.base_model.features[0][0] = nn.Conv2d(1, self.base_model.features[0][0].out_channels,
                                                   kernel_size=self.base_model.features[0][0].kernel_size,
                                                   stride=self.base_model.features[0][0].stride,
                                                   padding=self.base_model.features[0][0].padding,
                                                   bias=False)

        self.fc = nn.Linear(self.base_model.classifier[1].in_features, 1)
        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    def __init__(self, model_name='resnet18'):
        super(ResNet, self).__init__()
        if model_name == 'resnet18':
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet101':
            self.base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet152':
            self.base_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Invalid model name. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")

        # Modify the first convolutional layer to accept 1 channel input
        self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                          kernel_size=self.base_model.conv1.kernel_size,
                                          stride=self.base_model.conv1.stride,
                                          padding=self.base_model.conv1.padding,
                                          bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

    def forward(self, x):
        x = self.base_model(x)
        return x
