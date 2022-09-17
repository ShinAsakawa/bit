import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

class LeNet_Imagenet(nn.Module):
    """LeNet [@1998LeCun] の実装"""
    def __init__(self,
                 out_size:int=0,
                 device:str="cuda" if torch.cuda.is_available() else "cpu",
                ):
        super().__init__()

        # 第一畳み込み層の定義
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5,
                               device=device,
                              )

        # 最大値プーリング層の定義
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2).to(device=device)

        self.conv2 = nn.Conv2d(6, 16, 5).to(device=device) # 第二畳み込み層の定義

        # 第一全結合層の定義
        # ImageNet size であれば (((224 - 5 + 1) // 2) - 5 + 1) // 2 = 53
        self.fc1 = nn.Linear(in_features=16 * 53 * 53,
                             out_features=120).to(device=device)

        self.fc2 = nn.Linear(120, 84).to(device=device)      # 第二全結合層の定義
        self.fc3 = nn.Linear(84, out_size).to(device=device) # 最終層の定義

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#lenet = LeNet_Imagenet()
#y = lenet(torch.Tensor(np.array(img).transpose(2,0,1)))
#y = lenet(torch.randn(4, 3, 224, 224))
#print(y.size())
