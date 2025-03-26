# Source code and dataset for our paper "DBSFL: A Two-Stage Defense Method Against Adversarial Backdoor Attacks in Federated Learning"

## Basic Environment:
- Intel Core i5-12400F
- Python Version: 3.8.19
- PyTorch Version: 2.0.0

In addition, you can use pip to install basic packages such as numpy, matplotlib, pandas, pyautogui, and scapy.

## Source Code's Description:
In this section, we use the cifar-10 scenario as an example. You can download the corresponding dataset using the code in `main.py`.We store the called dataset in the dataset folder. 
The source code mainly consists of three parts:

- ### Model Definition：
We use the code in `FedNets.py` and `detect.py` to define the main task model and the detection model.

- ### Methods of defence
We store our defence method in the `attack.py`.

- ### Model training and testing
We store the specific training and testing processes in the `update-change.py`.
The code contains two parts: the main task model training and the detection model training. Please note that the paths in the code may not match your environment, so be sure to replace them accordingly.

## Quick start
cd main

## Citation
Coming soon
