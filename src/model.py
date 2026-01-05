# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. ResNet Block (1D)
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=False):
        super(ResBlock1D, self).__init__()
        self.downsample = downsample
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# 2. Attention Pooling
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2) # (Batch, Time, Feat)
        weights = self.attention(x)
        context = torch.sum(x * weights, dim=1)
        return context

# 3. Main Model (WITH FIX)
class CNNMultiLabelECG(nn.Module):
    def __init__(self, num_classes=9):
        super(CNNMultiLabelECG, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResBlock1D(64, 64, stride=1)
        self.layer2 = ResBlock1D(64, 128, stride=2, downsample=True)
        self.layer3 = ResBlock1D(128, 256, stride=2, downsample=True)
        self.layer4 = ResBlock1D(256, 512, stride=2, downsample=True)
        
        self.attention = AttentionPooling(512)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        
        # --- FIX: BIAS INITIALIZATION ---
        # Initialize bias to -4.0 (approx 0.018 probability)
        # This prevents the "All Zeros" problem in Epoch 1
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, -4.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        return self.classifier(x)

# 4. Focal Loss (Kept same)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()