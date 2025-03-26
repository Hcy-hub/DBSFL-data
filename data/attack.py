import copy

import hdbscan
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fftpack import dct
from sklearn.cluster import KMeans
from collections import defaultdict

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from scipy.spatial.distance import cosine

def krum(gradients, f):


    num_clients = len(gradients)
    distances = torch.zeros((num_clients, num_clients))


    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = 0
            for key in gradients[i]:
                dist += torch.norm(gradients[i][key] - gradients[j][key]) ** 2
            distances[i, j] = dist
            distances[j, i] = dist


    scores = torch.zeros(num_clients)
    for i in range(num_clients):
        sorted_distances, _ = torch.sort(distances[i])
        scores[i] = torch.sum(sorted_distances[:num_clients - f - 1])


    krum_index = torch.argmin(scores).item()
    print(krum_index)
    krum_gradient = gradients[krum_index]

    return krum_gradient


def multi_krum(gradients, f, m):

    num_clients = len(gradients)
    distances = torch.zeros((num_clients, num_clients))


    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = 0
            for key in gradients[i]:
                dist += torch.norm(gradients[i][key] - gradients[j][key]) ** 2
            distances[i, j] = dist
            distances[j, i] = dist


    scores = torch.zeros(num_clients)
    for i in range(num_clients):
        sorted_distances, _ = torch.sort(distances[i])
        scores[i] = torch.sum(sorted_distances[:num_clients - f - 1])


    _, indices = torch.topk(-scores, m, largest=True)
    selected_gradients = [gradients[i] for i in indices]


    multi_krum_gradient = {key: torch.zeros_like(gradients[0][key]) for key in gradients[0]}
    for grad in selected_gradients:
        for key in grad:
            multi_krum_gradient[key] += grad[key]
    for key in multi_krum_gradient:
        multi_krum_gradient[key] /= m

    return multi_krum_gradient



class FoolsGold:
    def __init__(self, num_clients, num_params):
        self.num_clients = num_clients
        self.memory = np.zeros((num_clients, num_params))
        self.weights = np.ones(num_clients)

    def update_memory(self, client_id, weights):
        self.memory[client_id] = weights



    def compute_cosine_similarity(self):

        norms = np.linalg.norm(self.memory, axis=1, keepdims=True)

        norms[norms == 0] = 1

        normalized_memory = self.memory / norms

        cosine_similarities = np.dot(normalized_memory, normalized_memory.T)

        np.fill_diagonal(cosine_similarities, 0)

        return cosine_similarities

    def compute_client_weights(self, cosine_similarities):
        max_similarities = np.max(cosine_similarities, axis=1)
        weights = 1 - max_similarities
        self.weights = np.maximum(weights, 0)
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights

    def aggregate_weights(self, client_weights):
        aggregate_weight = np.zeros_like(client_weights[0])
        for i, weights in enumerate(client_weights):
            # aggregate_weight += self.weights[i] * weights
            result = [x * self.weights[i] for x in weights]
            aggregate_weight = aggregate_weight + result

        return aggregate_weight

    def step(self, client_weights):
        for client_id, weights in enumerate(client_weights):
            self.update_memory(client_id, weights)

        cosine_similarities = self.compute_cosine_similarity()
        self.compute_client_weights(cosine_similarities)

        return self.aggregate_weights(client_weights)


import numpy as np
from scipy.spatial.distance import cdist




def flame(param_diffs, global_model, args, aggre_values_detect_list, net_detect):#something wrong

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_list = []


    param_diffs = [torch.tensor(diff) for diff in param_diffs]


    for i in range(len(param_diffs)):
        cos_i = []
        for j in range(len(param_diffs)):
            # param_diffs[i] 和 param_diffs[j]
            cos_ij = 1 - cos(param_diffs[i], param_diffs[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)


    num_clients = args.num_Chosenusers
    num_malicious_clients = args.attackers
    num_benign_clients = num_clients - num_malicious_clients


    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=num_clients // 2 + 1,
        min_samples=1,
        allow_single_cluster=True
    ).fit(cos_list)
    print(clusterer.labels_)


    benign_client = []
    norm_list = np.array([])
    max_num_in_cluster = 0
    max_cluster_index = 0

    if clusterer.labels_.max() < 0:
        for i in range(len(param_diffs)):
            benign_client.append(i)
            norm_list = np.append(norm_list, torch.norm(param_diffs[i], p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max() + 1):
            if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)

    for i in range(len(param_diffs)):
        norm_list = np.append(norm_list, torch.norm(param_diffs[i], p=2).item())



    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value / norm_list[i]
        if gama < 1:
            param_diffs[benign_client[i]] *= gama


    global_model = no_defence_balance([param_diffs[i] for i in benign_client], global_model)


    detect_updates = [aggre_values_detect_list[i] for i in benign_client]
    net_detect = no_defence_balance_list(detect_updates, net_detect)


    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0, std=args.noise * clip_value)
        var += temp

    return global_model, net_detect



def DBSFL(param_diffs, global_model, args, aggre_values_detect_list, net_detect):#Our defence method

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_list = []
    norm_list = []


    param_diffs = [torch.tensor(diff) for diff in param_diffs]


    for i in range(len(param_diffs)):
        cos_i = []
        for j in range(len(param_diffs)):

            cos_ij = 1 - cos(param_diffs[i], param_diffs[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)


    l2_norm_list = []
    for i in range(len(param_diffs)):
        l2_norm_list.append(torch.norm(param_diffs[i], p=2).item())


    clusterer_cos = hdbscan.HDBSCAN(
        min_cluster_size=args.num_Chosenusers // 2 + 1,
        min_samples=1,
        allow_single_cluster=True
    ).fit(cos_list)


    clusterer_l2 = hdbscan.HDBSCAN(
        min_cluster_size=args.num_Chosenusers // 2 + 1,
        min_samples=1,
        allow_single_cluster=True
    ).fit(np.array(l2_norm_list).reshape(-1, 1))

    print(f"cos clustering: {clusterer_cos.labels_}")
    print(f"L2 clustering: {clusterer_l2.labels_}")

    # 获取余弦相似度和L2范数的最大簇
    max_cos_cluster = np.argmax(np.bincount(clusterer_cos.labels_[clusterer_cos.labels_ >= 0]))
    max_l2_cluster = np.argmax(np.bincount(clusterer_l2.labels_[clusterer_l2.labels_ >= 0]))

    # 获取最大簇中的客户端
    benign_clients_cos = [i for i in range(len(param_diffs)) if clusterer_cos.labels_[i] == max_cos_cluster]
    benign_clients_l2 = [i for i in range(len(param_diffs)) if clusterer_l2.labels_[i] == max_l2_cluster]


    genuine_benign_clients = list(set(benign_clients_cos) & set(benign_clients_l2))


    norm_list = np.array(l2_norm_list)
    clip_value = np.median(norm_list[genuine_benign_clients])
    for i in genuine_benign_clients:
        gama = clip_value / norm_list[i]
        if gama < 1:
            param_diffs[i] *= gama


    global_model = no_defence_balance([param_diffs[i] for i in genuine_benign_clients], global_model)

    detect_updates = [aggre_values_detect_list[i] for i in genuine_benign_clients]  # 提取真正良性客户端的更新
    net_detect = no_defence_balance_list(detect_updates, net_detect)  # 更新检测网络



    return global_model, net_detect




def no_defence_balance(params, global_parameters):
    total_num = len(params)


    # sum_parameters = torch.zeros_like(params[0])
    sum_parameters = torch.zeros_like(params[0])
    for i in range(total_num):
        sum_parameters += params[i]


    averaged_parameters = sum_parameters / total_num
    averaged_parameters_dict = vector_to_parameters_dict(averaged_parameters, global_parameters)


    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            continue
        global_parameters[var] += averaged_parameters_dict[var]

    return global_parameters

def no_defence_balance_list(params, global_parameters):
    total_num = len(params)


    # sum_parameters = torch.zeros_like(params[0])
    sum_parameters = torch.zeros_like(torch.tensor(params[0]))
    for i in range(total_num):
        sum_parameters += torch.tensor(params[i])


    averaged_parameters = sum_parameters / total_num
    averaged_parameters_dict = vector_to_parameters_dict(averaged_parameters, global_parameters)


    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            continue
        global_parameters[var] += averaged_parameters_dict[var]

    return global_parameters


def vector_to_parameters_dict(vector, global_parameters):
    parameters_dict = {}
    start = 0

    for key, param in global_parameters.items():
        num_elements = param.numel()
        param_shape = param.shape


        parameters_dict[key] = vector[start:start + num_elements].view(param_shape)

        start += num_elements

    return parameters_dict


def identify_indicative_features(client_updates, n_clients, threshold=0.02):

    indicative_features = []
    client_appearance = defaultdict(int)


    num_params = len(client_updates[0])

    for param_index in range(num_params):

        param_updates = np.array([update[param_index] for update in client_updates])


        if np.all(param_updates == param_updates[0]):
            continue


        try:
            kmeans = KMeans(n_clusters=2)
            labels = kmeans.fit_predict(param_updates.reshape(-1, 1))
            centers = kmeans.cluster_centers_
            print(labels)
        except ValueError:
            continue

        distance = np.linalg.norm(centers[0] - centers[1])


        if distance > threshold:
            indicative_features.append(param_index)


            for client_idx, label in enumerate(labels):
                if label == 0:
                    client_appearance[client_idx] += 1

    return indicative_features, client_appearance


def identify_malicious_clients(client_appearance, indicative_feature_count, n_clients):

    malicious_clients = []
    for client_idx, appearance_count in client_appearance.items():
        if appearance_count > indicative_feature_count / 2:
            malicious_clients.append(client_idx)

    return malicious_clients


def auror_aggregation(client_updates, n_clients, threshold=0.02):

    indicative_features, client_appearance = identify_indicative_features(client_updates, n_clients, threshold)
    malicious_clients = identify_malicious_clients(client_appearance, len(indicative_features), n_clients)
    print(f"Identified malicious clients: {malicious_clients}")

    # 过滤掉恶意客户端的更新
    filtered_updates = [update for idx, update in enumerate(client_updates) if idx not in malicious_clients]

    # 平均化更新
    aggregated_update = np.mean(filtered_updates, axis=0).tolist()

    return aggregated_update



def apply_auror(global_model, client_updates, n_clients, threshold=0.02):
    aggregated_update = auror_aggregation(client_updates, n_clients, threshold)


    with torch.no_grad():
        for param, update in zip(global_model.parameters(), aggregated_update):
            param.add_(torch.tensor(update))

    return global_model


def FreqFed(w, param_diffs,values_glob,args):
    dct_lists = []
    dct_lists = [dct(sublist, type=2) for sublist in param_diffs]

    retain_ratio = 0.5
    trimmed_dct_lists = []

    for dct_list in dct_lists:

        trimmed = np.zeros_like(dct_list)

        num_to_retain = int(len(dct_list) * retain_ratio)


        trimmed[:num_to_retain] = dct_list[:num_to_retain]

        trimmed_dct_lists.append(trimmed)

    accepted_models = clustering(trimmed_dct_lists, args.num_Chosenusers)
    aggregated_result = aggregate_low_freq_components(param_diffs,accepted_models)


    return aggregated_result


def clustering(low_freq_components, clients):


    low_freq_components = np.array(low_freq_components)

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_list = []
    low_freq_components = [torch.tensor(low_freq_component) for low_freq_component in low_freq_components]


    for i in range(len(low_freq_components)):
        cos_i = []
        for j in range(len(low_freq_components)):
            if i == j:
                cos_ij = 0
            else:
                cos_ij = 1 - cos(low_freq_components[i], low_freq_components[j]).item()
            cos_i.append(cos_ij)
        cos_list.append(cos_i)



    clusterer = hdbscan.HDBSCAN(min_cluster_size=clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)
    cluster_ids = clusterer.labels_
    print(cluster_ids)


    unique, counts = np.unique(cluster_ids, return_counts=True)
    max_cluster = unique[np.argmax(counts)]


    B = [i for i in range(len(cluster_ids)) if cluster_ids[i] == max_cluster]
    print(B)


    return B


def aggregate_low_freq_components(low_freq_components, accepted_indices):


    selected_components = [low_freq_components[i] for i in accepted_indices]


    aggregated_result = np.mean(selected_components, axis=0)

    return aggregated_result