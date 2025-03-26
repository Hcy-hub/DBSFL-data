#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import ast
import os
from collections import OrderedDict

import pandas as pd
import torch
from torch.utils.data import Subset
import random
from torch import nn, autograd
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics
import copy
import pgd_like 
import torch.nn.functional as F

from FedNets import CNNMnistWithMask

import detect
from scipy.fftpack import dct


#matplotlib.use('Agg')


def poison_input(inputs, labels,trigger,mask, eval=False):
    bkd_ratio = 0.25
    target_class = 2
    if eval:
        bkd_num = inputs.shape[0]
    else:
        bkd_num = int(bkd_ratio * inputs.shape[0])
    inputs[:bkd_num] = trigger*mask + inputs[:bkd_num]*(1-mask)
    labels[:bkd_num] = target_class
    return inputs, labels, trigger


def poison_detect_input(inputs, labels, trigger, mask, eval=False):
    bkd_ratio = 0.5
    target_class = 1

    if eval:
        bkd_num = inputs.shape[0]
    else:
        bkd_num = int(bkd_ratio * inputs.shape[0])


    inputs[:bkd_num] = trigger + inputs[:bkd_num] * (1 - mask)
    labels[:bkd_num] = target_class


    labels[bkd_num:] = 0

    return inputs, labels, trigger





class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, tb, num, trigger, chosenUsers, adv_data):
        self.args = args
        self.tb = tb
        self.chosenUsers = chosenUsers
        self.adv_data = adv_data
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.NLLLoss()
        self.ldr_train, self.ldr_test = self.train_val_test(dataset, list(idxs))



    def train_val_test(self, dataset, idxs):
        # split train, and test
        idxs_train = idxs
        if (self.args.dataset == 'mnist') or (self.args.dataset == 'cifar') or (self.args.dataset == 'FashionMNIST') or (self.args.dataset == 'Adult'):
            idxs_test = idxs
            train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)
        else:
            train = self.args.dataset_train[idxs]
            test = self.args.dataset_test[idxs]
        return train, test

    def search_trigger(self, net,net_no_poison, dataset, trigger, mask):
        net_train = copy.deepcopy(net)
        net_train.eval()
        dm_adv_model_count = 1
        alpha = 0.01
        noise_loss_lambda = 0.01#0.01
        K=5 #5,20
        adv_models = []
        adv_ws = []
        ce_loss = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(net_train.parameters(),lr=self.args.lr,momentum=0.9)
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        # trigger.requires_grad_()
        for iter in range(K):
            if len(adv_models) > 0:
                for adv_model in adv_models:
                    del adv_model
            adv_models = []
            adv_ws = []
            for _ in range(dm_adv_model_count):  # 1
                adv_model, adv_w= self.get_adv_model(net_train,net_no_poison, dataset, trigger, mask)  # adv_w是平均相似度
                adv_models.append(adv_model)
            for batch_idx, (inputs, labels) in enumerate(dataset):
                count += 1
                inputs, labels, t = poison_input(inputs, labels, trigger, mask)
                outputs = net_train(inputs)
                loss = ce_loss(outputs, labels)

                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = noise_loss_lambda*adv_w*nm_loss/dm_adv_model_count
                        else:
                            loss += noise_loss_lambda*adv_w*nm_loss/dm_adv_model_count
                if loss != None:
                    loss.backward()
                    # optimizer.step()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min = -2, max = 2)
                    t.requires_grad_()
        # t = t.detach()
        trigger = t
        return trigger

    def get_adv_model(self, net,net_no_poison , dataset, trigger, mask):
        dm_adv_epochs = 1
        adv_model = copy.deepcopy(net)
        #prune part
        # p = net_prune.net_prune(self.args,adv_model, prune_rate=0.7)
        # adv_model = p.cnn_prune()
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.02, momentum=0.9, weight_decay=5e-4)
        for _ in range(dm_adv_epochs):
            for batch_idx, (inputs, labels) in enumerate(dataset):
                inputs,lbs,trigger = poison_input(inputs,labels, trigger, mask)
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(net_no_poison.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count




    def update_weights(self, net, num, trigger,mask, epoch, detect_net,selected_trigger, selected_masks,poison = False):
        lambda_constraint = 0.01#0.0001 0.01
        alpha = 0.01
        net_no_poison = copy.deepcopy(net)

        net_no_poison.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        optimizer = torch.optim.SGD(net_no_poison.parameters(), lr=self.args.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)  # Adam
        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.local_ep):  # 执行本地更新
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                    images, labels = autograd.Variable(images), \
                        autograd.Variable(labels)
                net_no_poison.zero_grad()
                log_probs = net_no_poison(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.gpu == -1:
                    loss = loss.cpu()
                self.tb.add_scalar('loss', loss.data.item())
                batch_loss.append(loss.data.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc, _, = self.test(net_no_poison, trigger, mask)
            epoch_acc.append(acc)
            if iter == 0:
                w_1st_ep = copy.deepcopy(net_no_poison.state_dict())

        # selected_trigger[num] = self.train_detect(detect_net,selected_trigger,selected_masks,10,num)
        selected_trigger = self.train_detect(detect_net,selected_trigger,selected_masks,10)



        if epoch > 20:
            if self.chosenUsers in [0,1]:
                if self.args.attack_ways == 'A3FL':
                    trigger = self.search_trigger(net, net_no_poison, self.ldr_train, trigger, mask)
                poison = True
                net_poison = copy.deepcopy(net_no_poison)
                net_poison.train()
                optimizer_poison = torch.optim.SGD(net_poison.parameters(),lr=self.args.lr,momentum=0.9)
                for iter in range(self.args.local_ep):
                    batch_loss_1 = []
                    for batch_idx,(images,labels) in enumerate(self.ldr_train):
                        if poison:
                            # images, labels = poison_input_TRAIN(images, labels, trigger, mask, eval=False)
                            if self.args.attack_ways == 'ADBA':
                                images, labels, t = poison_input(images, labels, trigger, mask[self.chosenUsers], eval=False)
                            else:
                                images, labels, t = poison_input(images, labels, trigger, mask, eval=False)
                        if self.args.gpu != -1:
                            images, labels = images.cuda(), labels.cuda()
                            images, labels = autograd.Variable(images), autograd.Variable(labels)
                        net_poison.zero_grad()
                        log_probs_poison = net_poison(images)
                        loss_poison = self.loss_func(log_probs_poison, labels)
                        loss_poison.backward()

                        #----------------------------------------------
                        #trigger updata
                        new_t = t - alpha*t.grad.sign()
                        t = new_t.detach_()
                        t = torch.clamp(t, min = -2, max = 2)
                        t.requires_grad_()
                        #------------------------------------------------

                        optimizer_poison.step()
                        if self.args.gpu == -1:
                            loss_poison = loss_poison.cpu()
                trigger = t


        avg_loss = sum(epoch_loss)/len(epoch_loss)
        avg_acc = sum(epoch_acc)/len(epoch_acc)


        if epoch > 20:
            if self.chosenUsers in [0, 1]:
                net = copy.deepcopy(net_poison)
            else:
                net = copy.deepcopy(net_no_poison)  # net_no_poison
        else:
            net = copy.deepcopy(net_no_poison)

        # if self.chosenUsers in [0,1]:
        #     net = copy.deepcopy(net_poison)
        # else:
        #     net = copy.deepcopy(net_no_poison)

        w = net.state_dict()

        gradient_info = {}
        for name,param in net.named_parameters():
            if param.grad is not None:
                gradient_info[name] = param.grad.detach()
        grad = gradient_info
        return w_1st_ep, w, avg_loss ,avg_acc,grad,trigger,self.adv_data,detect_net.state_dict(), selected_trigger
          

    def test(self, net,trigger,mask, poison = False):
        loss = 0
        log_probs = []
        labels = []
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            if poison:
                images , labels , trigger = poison_input(images,labels,trigger,mask,eval=True)
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            net = net.float()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
        if self.args.gpu != -1:
            loss = loss.cpu()
            log_probs = log_probs.cpu()
            labels = labels.cpu()
        y_pred = np.argmax(log_probs.data, axis=1)
        acc = metrics.accuracy_score(y_true=labels.data, y_pred=y_pred)
        loss = loss.data.item()         
        return acc, loss





    def test_backdoor(self, net, trigger, mask, poison, net_detect):
        loss = 0
        correct = 0
        total = 0
        target_label = 2
        malicious_count = 0
        total_count = 0
        total_data = 0

        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()


            non_target_indices = (labels != target_label)
            images = images[non_target_indices]
            labels = labels[non_target_indices]

            if len(labels) == 0:
                continue

            total_data += len(labels)


            if poison:
                images, labels, trigger = poison_input(images, labels, trigger, mask, eval=True)


            with torch.no_grad():
                images_dct = self.apply_grayscale_dct(images)
                malicious_probs = net_detect(images_dct.unsqueeze(1))
                y_pred_detect = np.argmax(malicious_probs,axis=1)#
                malicious_labels = y_pred_detect


            clean_indices = (malicious_labels == 0)
            images = images[clean_indices]
            labels = labels[clean_indices]


            if len(labels) == 0:
                return 0, 0, 1

            total_count += len(labels)
            malicious_count += (malicious_labels == 1).sum().item()



            images, labels = autograd.Variable(images), autograd.Variable(labels)
            net = net.float()
            log_probs = net(images)
            loss += self.loss_func(log_probs, labels).item()


            if self.args.gpu != -1:
                log_probs = log_probs.cpu()
                labels = labels.cpu()

            y_pred = np.argmax(log_probs.data, axis=1)


            correct += (y_pred == target_label).sum().item()
            total += y_pred.size(0)


        attack_success_rate = correct / total if total > 0 else 0
        avg_loss = loss / (batch_idx + 1)


        malicious_ratio = malicious_count / total_data if total_data > 0 else 0

        return attack_success_rate, avg_loss, malicious_ratio





    def apply_grayscale_dct(self,images):

        batch_size, channels, height, width = images.shape


        if channels == 3:  # RGB to Grayscale
            grayscale_images = (
                    0.2989 * images[:, 0, :, :] +
                    0.5870 * images[:, 1, :, :] +
                    0.1140 * images[:, 2, :, :]
            )
        elif channels == 1:  # Already Grayscale
            grayscale_images = images[:, 0, :, :]
        else:
            raise ValueError("Unsupported number of channels. Expect 1 (Grayscale) or 3 (RGB).")

        dct_images = torch.zeros_like(grayscale_images)
        for i in range(batch_size):
            # 应用 2D DCT
            dct_image = dct(dct(grayscale_images[i].numpy(), axis=1, norm='ortho'), axis=0, norm='ortho')
            dct_images[i] = torch.tensor(dct_image)

        return dct_images


    def train_detect(self, detect_net, chosen_trigger, selected_masks, num_epochs):
        optimizer = torch.optim.SGD(detect_net.parameters(), lr=self.args.lr, momentum=0.9)
        mask_detect_list = selected_masks

        for epoch in range(num_epochs):
            detect_net.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.ldr_train:
                batch_size = images.size(0)
                num_triggers = len(chosen_trigger)
                subset_size = batch_size // num_triggers


                poisoned_images = images.clone()
                poisoned_labels = labels.clone()

                for i, (trigger, mask) in enumerate(zip(chosen_trigger, mask_detect_list)):
                    start_idx = i * subset_size
                    end_idx = (i + 1) * subset_size if i < num_triggers - 1 else batch_size
                    with torch.no_grad():
                        poisoned_images[start_idx:end_idx], poisoned_labels[start_idx:end_idx], _ = poison_detect_input(
                            images[start_idx:end_idx], labels[start_idx:end_idx], trigger, mask
                        )


                dct_images = self.apply_grayscale_dct(poisoned_images)

                optimizer.zero_grad()
                # Forward pass with DCT-transformed images
                outputs = detect_net(dct_images.unsqueeze(1))
                loss = self.loss_func(outputs, poisoned_labels)
                # Backward pass
                loss.backward()

                for i in range(len(chosen_trigger)):
                    random_tensor = torch.randn_like(chosen_trigger[i])
                    sign_tensor = torch.sign(random_tensor)
                    perturbation = sign_tensor * 0.05
                    chosen_trigger[i] = chosen_trigger[i] + perturbation * mask_detect_list[i]

                optimizer.step()

        return chosen_trigger


