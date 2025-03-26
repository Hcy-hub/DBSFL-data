import torch
import torch.nn as nn
import torch.optim as optim

class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.fc = nn.Linear(32 * 26 * 26, 10)  # 26x26 is the result of 28x28 after Conv2d

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出 2 类，干净数据 or 对抗样本
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNForDetection(nn.Module):
    def __init__(self):
        super(CNNForDetection, self).__init__()


        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入单通道图像，输出32个通道
        self.pool = nn.MaxPool2d(2, 2)


        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)


        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)


        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (batch_size, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (batch_size, 64, 7, 7)
        x = self.pool(self.relu(self.conv3(x)))  # (batch_size, 128, 3, 3)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class SimpleCNNCifar(nn.Module):
    def __init__(self):
        super(SimpleCNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (batch_size, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (batch_size, 64, 7, 7)
        x = self.pool(self.relu(self.conv3(x)))  # (batch_size, 128, 3, 3)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn













import torch
import random


def split_trigger(adv_data, total_triggers):

    _, _, height, width = adv_data.shape


    sub_height = height // total_triggers
    sub_width = width // total_triggers

    sub_triggers = []
    sub_mask = []


    for i in range(total_triggers):
        for j in range(total_triggers):
            mask = torch.zeros_like(adv_data)
            start_h, end_h = i * sub_height, (i + 1) * sub_height
            start_w, end_w = j * sub_width, (j + 1) * sub_width
            mask[:, :, start_h:end_h, start_w:end_w] = 1
            sub_trigger = adv_data * mask
            sub_triggers.append(sub_trigger)
            sub_mask.append(mask)

    return sub_triggers, sub_mask


def select_random_triggers(sub_triggers, chosenusers, sub_mask):


    if chosenusers > len(sub_triggers):
        raise ValueError("chosenusers cannot be greater than the total number of sub-triggers.")


    selected_indices = random.sample(range(len(sub_triggers)), chosenusers)

    selected_triggers = [sub_triggers[i] for i in selected_indices]
    selected_masks = [sub_mask[i] for i in selected_indices]

    return selected_triggers, selected_masks



