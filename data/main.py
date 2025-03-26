#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from operator import length_hint
import time
import matplotlib
import sys

import matplotlib.pyplot as plt
import pylab
from scipy.optimize import minimize
from sklearn.cluster import KMeans
#plt.switch_backend('agg')
#matplotlib.use('Agg')
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import copy
import pandas as pd
import math
import numpy as np
import random
import collections
import torch
import torch.nn.functional as F
import sympy as sy
from numpy import linalg as la
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from tqdm import tqdm
from torch import autograd, optim
from tensorboardX import SummaryWriter
from sympy import solve
from sympy.abc import P, y
from scipy import optimize
from collections import Counter
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from torch.utils.data import Subset

from hh import huffman_encode, huffman_decode
from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid,fashion_mnist_iid,fashion_mnist_noniid,adult_iid,adult_noniid
from options import args_parser
from Update_change import LocalUpdate
from FedNets import MLP1, CNNMnist, CNN_test, CNNCifar, MLP3, MLP_triple ,MLP_regression,CNNFashionMnist,VGG
from averaging import average_weights, average_weights_orig
from Calculate import minkowski_distance, mahala_distance, noise_add, sample_para
import pickle
from csvec import CSVec
from sklearn.model_selection import train_test_split
import argparse
import attack
from textwrap import indent

import torch.nn.utils.prune as prune

import detect

def value_replace_2(w, value_sequence):
    w_rel = copy.deepcopy(w)
    m =0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m =m +1
    return w_rel




def net_grad_update_krum(w, values_glob, aggre_values_increment_list):
    w_rel = copy.deepcopy(w)
    arrays_dict_values = {}
    client_list_values = []
    extracted_arrays_values = [element for element in aggre_values_increment_list]
    for i, client in enumerate(extracted_arrays_values):
        arrays_dict_values[f'client_{i}'] = client
    for key,value in arrays_dict_values.items():
        change_client = value
        n = 0

        w_rel_copy = collections.OrderedDict((k, torch.zeros_like(v)) for k, v in w_rel.items())
        for layer_name, tensor in w_rel_copy.items():

            shape = tensor.shape

            size = tensor.numel()
            reshaped_tensor = np.array(change_client[n:n + size]).reshape(shape)

            w_rel_copy[layer_name] = torch.tensor(reshaped_tensor)
            n += size

        client_list_values.append(w_rel_copy)
    krum_weight = attack.krum(client_list_values, f=1)
    weight_increase = []
    for i in krum_weight.keys():
        weight_increase += list(krum_weight[i].view(-1).cpu().numpy())
    weight_increase = np.array(weight_increase)
    m = 0
    value_sequence = weight_increase + values_glob
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m = m+1
    return w_rel

def net_grad_update_multi_krum(w, values_glob, aggre_values_increment_list):
    w_rel = copy.deepcopy(w)
    arrays_dict_values = {}
    client_list_values = []
    extracted_arrays_values = [element for element in aggre_values_increment_list]
    for i, client in enumerate(extracted_arrays_values):
        arrays_dict_values[f'client_{i}'] = client
    for key,value in arrays_dict_values.items():
        change_client = value
        n = 0

        w_rel_copy = collections.OrderedDict((k, torch.zeros_like(v)) for k, v in w_rel.items())
        for layer_name, tensor in w_rel_copy.items():

            shape = tensor.shape

            size = tensor.numel()
            reshaped_tensor = np.array(change_client[n:n + size]).reshape(shape)

            w_rel_copy[layer_name] = torch.tensor(reshaped_tensor)
            n += size

        client_list_values.append(w_rel_copy)
    krum_weight = attack.multi_krum(client_list_values, f=1,m=3)
    weight_increase = []
    for i in krum_weight.keys():
        weight_increase += list(krum_weight[i].view(-1).cpu().numpy())
    weight_increase = np.array(weight_increase)
    m = 0
    value_sequence = weight_increase + values_glob
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m = m+1
    return w_rel

def net_grad_update_foolsgold(w, values_glob,num_clients, net_glob, weight):
    w_rel = copy.deepcopy(w)
    num_params = sum(p.numel() for p in net_glob.parameters())
    fg = attack.FoolsGold(num_clients,num_params)
    client_weights = weight
    aggregated_weight = fg.step(client_weights)
    weight_increase = np.array(aggregated_weight)
    value_sequence = weight_increase + values_glob
    m = 0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m = m+1
    return w_rel


def net_grad_update_DBSFL(param_diffs, global_model, args, aggre_values_detect_list,net_detect):
    w_rel , detect_rel= attack.DBSFL(param_diffs, global_model, args, aggre_values_detect_list,net_detect)
    return w_rel, detect_rel


def net_grad_update_FreqFed(w, param_diffs,values_glob,args):
    w_rel = copy.deepcopy(w)
    value_sequence = attack.FreqFed(w, param_diffs,values_glob,args)
    value_sequence = value_sequence + values_glob
    m = 0

    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m = m +1

    return w_rel



def net_grad_update_auror(global_model, client_updates, n_clients):
    global_model = attack.apply_auror(global_model,client_updates, n_clients)
    return global_model





def value_replace_diff(w, value_sequence,values_glob,grad_increment_list):
    w_rel = copy.deepcopy(w)
    m =0
    value_sequence = value_sequence + values_glob


    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m = m +1
    return w_rel















if __name__ == '__main__': 

    # return the available GPU
    """
    av_GPU = torch.cuda.is_available()
    if  av_GPU == False:
        exit('No available GPU')
    print("\n GPU is running !!! \n ")
    """
    
    run_start_time = time.time() 
    # parse args

    #args = args_parser()
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='0,1', type=str, help='choise gpu')
    parser.add_argument('--no_cuda', action='store_true', help='no gpu')

    args = parser.parse_args(args=['--device', '0',  '--no_cuda'])


    # define paths
    path_project = os.path.abspath('..')
    

    summary = SummaryWriter('local')


    args.device = "cpu"
    
    args.num_cols = 500
    args.num_rows = 500
    args.num_blocks = 1
    args.grad_size =0
    args.weight_decay =1
    args.max_grad_norm = 1e5
    args.num_workers =10

    args.gpu = -1              # -1 (CPU only) or GPU = 0
    args.lr = 0.02            # 0.001 for cifar dataset  0.02 mlp
    args.model = 'cnn'
    args.dataset = 'cifar'


    args.num_users = 10        # numb of users cautious : the number must more than 10
    args.num_Chosenusers = 8
    args.attackers = 1
    args.epochs = 400        # numb of global iters
    args.local_ep = 3         # numb of local iters
    args.num_experiments = 1


    args.num_items_train = 1200 # numb of local data size # local image date numbers minist int[(60000)/users] cifar 50000
    args.num_items_test =  500 # local image date test cifar 10000/num_users  mnist 800
    args.local_bs = 500         # Local Batch size (1200 = full dataset)
                               # size of a user for mnist, 2000 for cifar)
    
                               
    args.set_epochs = [10]


    args.degree_noniid = 0
    args.set_degree_noniid = [0]
    args.strict_iid = True#
    args.iid = True

    
    
    args.set_momentum = False  # nedd_add_momentum = True   no_add_momentum = False
    args.momentum_beta= 0.5    # set momentum_beta = 1-w
    momentum_beta = args.momentum_beta                          

    args.quantile_level = 2
    args.set_quantile_level = [2]
    quantile_level = 2
    args.ratio_train = [1,1,1,1,1]

    
    args.parameter_ways = 'diff_parameter'  #  'orig_parameter' , 'diff_parameter'
    # args.set_sketch_sche = ['count_sketch','kmeans_opt','uniform_quantile_opt_propose','quantile_bucket_opt_proposed','QSGD','bucket_quantile']
    args.set_sketch_sche = ['orig_1'] #  'orig_1' 'orig' 'SVD_Split' 'count_sketch_news'，'kmeans_opt','kmeans_opt8','kmeans_opt16',STC,kmeans_opt_huffman

    args.sketch_sche ='orig_1'
    args.defense_ways = ('DBSFL') #'krum' multi_krum none foolsgold RFA flame auror FreqFed
    args.attack_ways = 'ADBA'#ADBA,A3FL


    args.num_L = 2
    args.delta = 1e-5
    args.ss = 3705
    args.num_classes =10

    #源代码中的flame
    args.wrong_mal = 0
    args.right_ben = 0
    args.turn = 0
    args.noise = 0.001

    args.alpha = 0.2
    args.log_root = 'logs/'
    args.pruning_by = 'number'#number,threshold
    args.pruning_max = 2000 #0.90
    args.pruning_step = 1#0.05




    args.set_variable = args.set_degree_noniid
    args.set_variable0 = copy.deepcopy(args.set_quantile_level)
    args.set_variable1 = copy.deepcopy(args.set_sketch_sche)

    hash_deepth =5

    batch_size = 1024

    adv_data = torch.ones((1, 3, 32, 32), requires_grad=False,
                          device='cpu') * 0.5
    adv_data.requires_grad_()
    sub_triggers, sub_mask = detect.split_trigger(adv_data, args.num_Chosenusers)#chosenusers * chosenusers

    if args.attack_ways == 'A3FL':
        trigger = torch.ones((1, 3, 32, 32), requires_grad=False,
                             device='cpu') * 0.5
        mask = torch.zeros_like(trigger)
        mask[:, :, 12:20, 12:20] = 1
    elif args.attack_ways == 'ADBA':
        trigger = torch.ones((1, 3, 32, 32), requires_grad=False,
                            device='cpu') * 0.5
        mask = []
        mask_true = torch.zeros_like(trigger)
        mask_1 = mask_true
        mask_1[:,:,12:16,12:16] = 1
        mask_2 = mask_true
        mask_2[:,:,16:20,16:20] = 1
        mask_true[:, :, 12:20, 12:20] = 1
        mask = [mask_1,mask_2,mask_true]

    trigger.requires_grad_()

    print('--------------- information------------\n ways =',args.set_sketch_sche)
    print('          quantile_level    = ',args.set_quantile_level)
    print('          learning rate     = ',args.lr)
    print('          num_experiments   = ',args.num_experiments)
    print('          global_epochs     = ',args.epochs)
    print('          date_set          = ',args.dataset)
    print('          model             = ',args.model)
    print('          parameter_ways    = ',args.parameter_ways)
    print('          set_degree_noniid = ',args.set_degree_noniid)
    print('          args.iid          = ',args.iid)
    print('          args.set_momentum = ',args.set_momentum)
    if args.set_momentum == True :
        print('          args.momentum_beta= ',args.momentum_beta)
    print('----------------information------------\n')
    

    apply_transform1 = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    apply_transform2 = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    

    
    if not os.path.exists('./experiresult'):
        os.mkdir('./experiresult')



    # load dataset and split users

    dict_users,dict_users_train,dict_users_test = {},{},{}
    dataset_train,dataset_test = [],[]

    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./dataset/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
        dataset_test = datasets.MNIST('./dataset/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
            # sample users
        if args.iid:
            dict_users = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)

        else:
            dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
        
    elif args.dataset == 'cifar':
        dict_users_train, dict_sever = {},{}
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./dataset/cifar/', train=True, transform=transform, target_transform=None, download=True)
        dataset_test = datasets.CIFAR10('./dataset/cifar/', train=False, transform=transform, target_transform=None, download=True)
        # train_data =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)#实验
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.num_items_train)#实验
            dict_sever = cifar_iid(dataset_test, args.num_users, args.num_items_test)#训练
            """
            num_train = int(0.6*args.num_items_train)
            for idx in range(args.num_users):
                dict_users_train[idx] = set(list(dict_users[idx])[:num_train])
                dict_sever[idx] = set(list(dict_users[idx])[num_train:])
            """
        else:
            dict_users = cifar_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = cifar_noniid(args, dataset_test, args.num_users, args.num_items_test)
            """
            dict_test = []
            num_train = int(0.6*args.num_items_train)
            for idx in range(args.num_users):
                dict_users_train[idx] = set(list(dict_users[idx])[:num_train])
                dict_sever[idx] = set(list(dict_users[idx])[num_train:])
            """





    img_size = dataset_train[0][0].shape



    print('                       quantile_level = ',quantile_level)
    for v in range(len(args.set_variable)):
        final_train_loss = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_train_accuracy = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_test_loss = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_test_accuracy = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_quantile_err = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        args.degree_noniid = copy.deepcopy(args.set_variable[v])
        for s in range(len(args.set_variable0)):
            timeslot = time.time()
            test_acc_record, test_loss_record, quantile_err_record,test_asr_record,test_malicious_ratio_record = [], [], [], [], []
            percent_quantile_err_record,test_weight_sum_record =[],[]
            for j in range(len(args.set_variable1)):
                args.sketch_sche = copy.deepcopy(args.set_variable1[j])
                args.quantile_level = copy.deepcopy(args.set_variable0[s])
                quantile_level = args.quantile_level
                loss_test, loss_train = [], []
                acc_test, acc_train = [], [] 
                com_cons = []
                fin_loss_test_list = []
                fin_acc_test_list = []
                fin_asr_test_list = []
                fin_malicious_ratio_test_list = []
                fin_quantile_err = []
                fin_expr_weights_sum_avg_list,fin_expr_percent_quantile_err_avg_list =[],[]

                for m in range(args.num_experiments):
                    net_glob = None
                    if args.model == 'cnn' and args.dataset == 'mnist':
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            net_glob = CNN_test(args=args).cuda()
                        else:
                            net_glob = CNNMnist(args=args)
                    elif args.model == 'mlp' and args.dataset == 'mnist':
                        len_in = 1
                        for x in img_size:
                            len_in *= x
                        print('\n  mlp dim_in = ',len_in)
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).cuda()
                        else:
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes)
                    elif args.model == 'cnn' and args.dataset == 'cifar':
                        if args.gpu != -1:
                            net_glob = CNNCifar(args).cuda()
                        else:
                            net_glob = CNNCifar(args)



                    net_detect = detect.SimpleCNNCifar()#detect model choise
                    net_detect.train()


                

                    net_glob.train()  #Train() does not change the weight values
                    # copy weights

                    w_glob = net_glob.state_dict()
                    values_glob =[]
                    for i in w_glob.keys():
                            values_glob += list(w_glob[i].view(-1).cpu().numpy())
                    if args.set_momentum == True :
                        mountum_t_0 = [0 for i in range(len(values_glob))]

                    one_expr_train_loss_avg_list, one_expr_train_acc_avg_list, one_expr_test_loss_avg_list, one_expr_test_acc_avg_list, one_expr_quantile_err_avg_list,one_expr_test_asr_avg_list,one_expr_test_malicious_ratio_list = [], [], [], [], [], [], []
                    one_expr_weights_sum_avg_list,one_expr_percent_quantile_err_avg_list =[],[]
                    quantile_err = 0
                    communciation_lr = []
                    for iter in range(args.epochs):

                        print('\n','*' * 20,f'Experiment: {m}/{args.num_experiments}, Epoch: {iter}/{args.epochs}','*' * 20)
                        #print('                       quantile_level = ',quantile_level)
                        time_start = time.time() 
                        if  args.num_Chosenusers < args.num_users:
                            chosenUsers = random.sample(range(args.num_users),args.num_Chosenusers)
                            chosenUsers.sort()
                            #子触发器
                            chosen_trigger, selected_masks = detect.select_random_triggers(sub_triggers, args.num_Chosenusers  , sub_mask)#args.num_Chosenusers

                            print(chosenUsers)
                        else:
                            chosenUsers = range(args.num_users)
                        #print("\nChosen users:", chosenUsers)
                        print('\nsketch ways = ',args.sketch_sche)                
                        w_locals, w_locals_1ep, train_loss_locals_list, train_acc_locals_list = [], [], [], []
                        quantile_err_list = []
                        weights_sum_list,percent_quantile_err_list = [],[]
                        w_glob = net_glob.state_dict()
                        values_glob = []
                        values_grad_glob = []
                        for i in w_glob.keys():
                            values_glob += list(w_glob[i].view(-1).cpu().numpy())


                        w_detect = net_detect.state_dict()
                        values_detect = []
                        for i in w_detect.keys():
                            values_detect += list(w_detect[i].view(-1).cpu().numpy())
                        values_detect_list = []

                        
                        values_increment_list = []

                        grad_increment_list = []

                        communciation_singal_lr =[]

                        adv_data_list = []

                        for idx in range(len(chosenUsers)):

                                #------------------------------------------------------------------
                                # portion = 0.1
                                # n_backdoor = int(portion * len(dataset_train))
                                # target_label = 0
                                # backdoor_indices = np.random.choice(len(dataset_train), n_backdoor, replace=False)
                                # for m in backdoor_indices:
                                #     data, target = dataset_train[m]
                                #     data_change = add_trigger(data)
                                #     dataset_train.data[m] = data_change
                                #     dataset_train.targets[m] = target_label


                            # adv_data_copy = adv_data.clone().detach()
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]], tb=summary, num=idx, trigger=trigger, chosenUsers=chosenUsers[idx], adv_data= adv_data)
                            # print(local.ldr_train.dataset)
                            # print(dataset_train
                            # print(net_glob.state_dict().keys())

                            w_1st_ep, w, loss, acc,grad,trigger,adv_data_local,net_detect_local, chosen_trigger = local.update_weights(net=copy.deepcopy(net_glob),num=idx,trigger=trigger,mask=mask,epoch = iter, detect_net = copy.deepcopy(net_detect),selected_trigger=chosen_trigger, selected_masks=selected_masks,poison=False)
                            train_loss_locals_list.append(copy.deepcopy(loss))
                            train_acc_locals_list.append(copy.deepcopy(acc))

                            detect_increase = copy.deepcopy(net_detect_local)


                            
                            w_increas = copy.deepcopy(w)

                            grad_values = copy.deepcopy(grad)

                            values_increment = []

                            detect_increment = []

                            grad_increment = []#in this way grad_increment is grad

                            if args.parameter_ways == 'orig_parameter' :   #  'orig_parameter' diff_parameter
                                for i in w_increas.keys():
                                    values_increment += list(w_increas[i].view(-1).cpu().numpy())
                                for i in grad_values.keys():
                                    grad_increment += list(grad_values[i].view(-1).cpu().numpy())
                            elif args.parameter_ways == 'diff_parameter' : #  'orig_parameter' diff_parameter
                                for i in w_increas.keys():
                                    values_increment += list(w_increas[i].view(-1).cpu().numpy()-w_glob[i].view(-1).cpu().numpy())
                                for i in grad_values:
                                    grad_increment += list(grad_values[i].view(-1).cpu().numpy())
                                for i in detect_increase.keys():
                                    detect_increment += list(detect_increase[i].view(-1).cpu().numpy()-w_detect[i].view(-1).cpu().numpy())

                            
                            weights_sum = sum(abs(np.array(values_increment)))


                            if args.sketch_sche == 'orig':
                                orig_values_increment = copy.deepcopy(values_increment) 
                                print("model length =  ",len(orig_values_increment))
                                w_weights = sum(abs(np.array(orig_values_increment)))
                                print("w_weights = ",w_weights)

                            elif args.sketch_sche == 'orig_1':
                                orig_values_increment = copy.deepcopy(values_increment)
                                print("model length =  ",len(orig_values_increment))
                                w_weights = sum(abs(np.array(orig_values_increment)))
                                print("w_weights = ",w_weights)



                            quantile_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
                            quantile_err_list.append(quantile_err)

                            percent_quantile_err = 100*quantile_err/weights_sum

                            weights_sum_list.append(weights_sum)
                            percent_quantile_err_list.append(percent_quantile_err)
                            values_increment_list.append(values_increment)

                            grad_increment_list.append(grad_increment)

                            values_detect_list.append(detect_increment)

                            # adv_data_list.append(adv_data_local)

                        aggre_values_increment_list = values_increment_list
                        aggre_grad_increment_list = grad_increment_list
                        aggre_values_detect_list = values_detect_list

                        values_increment_list = np.sum(values_increment_list,axis = 0)
                        values_increment_list = values_increment_list/args.num_Chosenusers


                        # sum_adv_data_list = sum(adv_data_list)
                        # adv_data = sum_adv_data_list/len(adv_data_list)


                        grad_increment_list = np.sum(grad_increment_list,axis = 0)
                        grad_increment_list = grad_increment_list/args.num_Chosenusers

                        # grad_increment_list = [x**2 for x in grad_increment_list






                        communciation_lr.append(np.sum(communciation_singal_lr,axis = 0)/args.num_Chosenusers)
                        #set_momentum
                        if args.set_momentum == True : 
                            values_increment_list = momentum_beta* np.array(mountum_t_0) + (1-momentum_beta)* np.array(values_increment_list)
                            mountum_t_0 =values_increment_list

                        if args.defense_ways == 'none':
                            if args.parameter_ways == 'orig_parameter':  # 'orig_parameter' diff_parameter
                                w_glob = value_replace_2(w_increas, values_increment_list)
                            elif args.parameter_ways == 'diff_parameter':  # 'orig_parameter' diff_parameter
                                w_glob = value_replace_diff(w_increas, values_increment_list, values_glob,
                                                                          grad_increment_list)
                            # copy weight to net_glob
                            net_glob.load_state_dict(w_glob)

                        if args.defense_ways == 'krum':
                            w_glob = net_grad_update_krum(w_increas,
                                                               values_glob, aggre_values_increment_list)
                            # w_glob = net_grad_update_krum_text(w_increas,aggre_grad_increment_list, args.lr, values_glob,aggre_values_increment_list)
                            net_glob.load_state_dict(w_glob)
                        if args.defense_ways == 'multi_krum':
                            w_glob = net_grad_update_multi_krum(w_increas,values_glob, aggre_values_increment_list)
                            net_glob.load_state_dict(w_glob)

                        if args.defense_ways == 'foolsgold':
                            w_glob = net_grad_update_foolsgold(w_increas,values_glob,args.num_Chosenusers , net_glob, aggre_values_increment_list)
                            net_glob.load_state_dict(w_glob)
                        # if args.defense_ways == 'flame':
                        #     w_glob = net_grad_update_flame_1(net_glob, args.num_Chosenusers, args.num_L, args.delta, aggre_values_increment_list, args.ss)
                        #     net_glob.load_state_dict(w_glob.state_dict())
                        if args.defense_ways == 'DBSFL':
                            w_glob , w_detect= net_grad_update_DBSFL(aggre_values_increment_list, net_glob.state_dict(), args, aggre_values_detect_list, net_detect.state_dict())
                            net_glob.load_state_dict(w_glob)
                            net_detect.load_state_dict(w_detect)

                        if args.defense_ways == 'auror':
                            net_glob = net_grad_update_auror(net_glob, aggre_values_increment_list, args.num_Chosenusers)
                        if args.defense_ways == 'FreqFed':
                            w_glob = net_grad_update_FreqFed(w_increas,aggre_values_increment_list, values_glob, args)
                            net_glob.load_state_dict(w_glob)





                        # global test
                        test_loss_locals_list, test_acc_locals_list, test_asr_list ,test_malicious_ratio_list= [], [], [], []
                        net_glob.eval()
                        net_detect.eval()

                        for c in range(args.num_users):
                            net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_sever[idx], tb=summary,
                                                    num=idx, trigger=trigger, chosenUsers=chosenUsers[idx],
                                                    adv_data=adv_data)
                            if args.attack_ways == 'A3FL':
                                acc, loss = net_local.test(net=net_glob, trigger=trigger, mask=mask)
                                bkd_acc, bkd_loss ,malicious_ratio= net_local.test_backdoor(net=net_glob, trigger=trigger, mask=mask,
                                                                            poison=True,net_detect=net_detect)
                            elif args.attack_ways == 'ADBA':
                                acc, loss = net_local.test(net=net_glob, trigger=trigger, mask=mask[2])
                                bkd_acc, bkd_loss, malicious_ratio = net_local.test_backdoor(net=net_glob,
                                                                                             trigger=trigger,
                                                                                             mask=mask[2],
                                                                                             poison=True,
                                                                                             net_detect=net_detect)
                            test_asr_list.append(bkd_acc)
                            test_acc_locals_list.append(acc)
                            test_loss_locals_list.append(loss)
                            test_malicious_ratio_list.append(malicious_ratio)

                        train_loss_avg = sum(train_loss_locals_list) / len(train_loss_locals_list)
                        train_acc_avg = sum(train_acc_locals_list) / len(train_acc_locals_list)

                        test_loss_avg = sum(test_loss_locals_list) / len(test_loss_locals_list)
                        test_acc_avg = sum(test_acc_locals_list) / len(test_acc_locals_list)  # 11
                        test_asr_avg = sum(test_asr_list) / len(test_asr_list)
                        test_malicious_ratio_avg = sum(test_malicious_ratio_list) / len(test_malicious_ratio_list)
                        quantile_err_avg = sum(quantile_err_list) / len(quantile_err_list)

                        weights_sum_avg = sum(weights_sum_list) / len(weights_sum_list)
                        percent_quantile_err_avg = sum(percent_quantile_err_list) / len(percent_quantile_err_list)

                        one_expr_train_loss_avg_list.append(train_loss_avg)
                        one_expr_train_acc_avg_list.append(train_acc_avg)

                        one_expr_test_loss_avg_list.append(test_loss_avg)
                        one_expr_test_acc_avg_list.append(test_acc_avg)
                        one_expr_test_asr_avg_list.append(test_asr_avg)
                        one_expr_test_malicious_ratio_list.append(test_malicious_ratio_avg)
                        one_expr_quantile_err_avg_list.append(quantile_err_avg)
                        one_expr_weights_sum_avg_list.append(weights_sum_avg)
                        one_expr_percent_quantile_err_avg_list.append(percent_quantile_err_avg)
                        time_end = time.time()
                        print("\nTrain acc: {} Test acc:  {}".format(train_acc_avg, test_acc_avg))
                        print("\nTest loss:{}".format(test_loss_avg))
                        print("\nasr: {} bkd_loss: {}".format(bkd_acc, bkd_loss))
                        print("\nmalicious_ratio: {} ".format(malicious_ratio))

                    fin_loss_test_list.append(one_expr_test_loss_avg_list)
                    fin_acc_test_list.append(one_expr_test_acc_avg_list)
                    fin_asr_test_list.append(one_expr_test_asr_avg_list)
                    fin_malicious_ratio_test_list.append(one_expr_test_malicious_ratio_list)
                    fin_quantile_err.append(one_expr_quantile_err_avg_list)

                    fin_expr_weights_sum_avg_list.append(one_expr_weights_sum_avg_list)
                    fin_expr_percent_quantile_err_avg_list.append(one_expr_percent_quantile_err_avg_list)

                tmp_fin_mean_loss_test = np.array(fin_loss_test_list).mean(axis=0)
                tmp_fin_mean_acc_test = np.array(fin_acc_test_list).mean(axis=0)
                tmp_fin_mean_asr_test = np.array(fin_asr_test_list).mean(axis=0)
                tmp_fin_mean_malicious_ratio_test = np.array(fin_malicious_ratio_test_list).mean(axis=0)
                tmp_fin_mean_quantile_test = np.array(fin_quantile_err).mean(axis=0)

                tmp_fin_mean_weights_sum_test = np.array(fin_expr_weights_sum_avg_list).mean(axis=0)
                tmp_fin_mean_percent_quantile_err_test = np.array(fin_expr_percent_quantile_err_avg_list).mean(axis=0)

                test_loss_record.append(tmp_fin_mean_loss_test.tolist())
                test_acc_record.append(tmp_fin_mean_acc_test.tolist())
                test_asr_record.append(tmp_fin_mean_asr_test.tolist())
                test_malicious_ratio_record.append(tmp_fin_mean_malicious_ratio_test.tolist())
                quantile_err_record.append(tmp_fin_mean_quantile_test.tolist())
                test_weight_sum_record.append(tmp_fin_mean_weights_sum_test.tolist())
                percent_quantile_err_record.append(tmp_fin_mean_percent_quantile_err_test.tolist())
                print('\n\n------------------- all ways record : -----------------')
                print('test_loss_record:', test_loss_record)
                print('test_acc_record:', test_acc_record)
                print('test_asr_record:', test_asr_record)
                print('test_malicious_record:', test_malicious_ratio_record)

                x = [i for i in range(args.epochs)]
                # plot acc curve
                labels = "quantile level-{}".format(args.quantile_level)

                plt.figure(m)
                plt.title('lr = {}'.format(args.lr), fontsize=20)
                for i in range(len(test_asr_record)):
                    labels = "{}".format(args.set_sketch_sche[i])
                    plt.plot(x, test_asr_record[i], label=labels)
                plt.legend(loc='upper left', bbox_to_anchor=(1.0, 0))
                plt.ylabel('test asr')
                plt.xlabel('epoches')
                plt.grid(linestyle="--")

                plt.savefig('./experiresult/{}-quantile_level-{}training_record-{}-{}.pdf'. \
                            format(args.sketch_sche, quantile_level, iter + 1, timeslot))


            plt.close()
            print('\n\n expriment finished')
            run_end_time = time.time()
            print('\n Experiment Run time = {} h'.format((run_end_time - run_start_time) / 3600))
