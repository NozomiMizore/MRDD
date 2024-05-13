import sys
# 3.8 supported
# from math import prod

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans
from munkres import Munkres


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    首先将输出的概率值按照指定的 k 进行排序，然后与真实标签比较，计算每个 k 的准确率并返回
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Using either a newly instantiated ClusterClass or a provided cluster_obj, generates
        cluster assignments based on input data.

        Args:
            x: the points with which to perform clustering
            cluster_obj: a pre-fitted instance of a clustering class 拟合好的聚类对象，
            如果为 None, 则需要通过 ClusterClass 和 n_clusters 实例化新的聚类对象。
            ClusterClass: a reference to the sklearn clustering class, necessary
              if instantiating a new clustering class
            n_clusters: number of clusters in the dataset, necessary
                        if instantiating new clustering class
            init_args: any initialization arguments passed to ClusterClass

        Returns:
            a tuple containing the label assignments and the clustering object
    """
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def clustering_metric(y_true, y_pred, decimals=4):
    """Get clustering metric"""
    n_clusters = np.size(np.unique(y_true))
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)
    
    class_acc, p, fscore = classification_metric(y_true, y_pred_ajusted)
    
    # ACC
    acc = clustering_accuracy(y_true, y_pred)
    acc = np.round(acc, decimals)
    
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return acc, nmi, ari, class_acc, p, fscore


def clustering_accuracy(y_true, y_pred):
    """
    首先根据真实标签和预测标签创建混淆矩阵，然后利用匈牙利算法进行最优匹配，最终计算准确率并返回。
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # 混淆矩阵
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # 使用线性求和分配算法（匈牙利算法）来找到最优匹配。
    # 这里通过求最大权重减去混淆矩阵中的权重，以将最大化匹配问题转换为最小化问题。
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

    # 计算匹配结果的总权重，并将其除以预测标签的大小，以计算聚类的准确率。
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """Get classification metric"""
   
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return accuracy, precision, f_score


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Computes the predicted labels, where label assignments now
        correspond to the actual labels in y_true (as estimated by Munkres)

        Args:
            cluster_assignments: array of labels, outputted by kmeans
            y_true:              true labels
            n_clusters:          number of clusters in the dataset

        Returns:
            a tuple containing the accuracy and confusion matrix,
                in that order
    """
    # 计算真实标签和聚类标签之间的混淆矩阵
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    # 计算混淆矩阵的成本矩阵，并使用 Munkres 算法计算最优的匹配
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    # 根据 Munkres 算法得到的最优匹配，获取将 KMeans 聚类标签映射到真实标签的标签映射
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    # 如果聚类标签中的最小值不为零，则将所有聚类标签减去最小值，以确保它们从零开始
    # 然后，根据 KMeans 聚类标签映射到真实标签的映射，得到最终的预测标签
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def calculate_cost_matrix(C, n_clusters):
    """
    根据混淆矩阵计算了每个簇分配给每个标签的代价，并返回代价矩阵
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    """
    根据最优匹配的索引创建簇与真实标签的映射，并返回标签列表
    """
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def clustering_by_representation(X_rep, y):
    """Get scores of clustering by representation"""
    n_clusters = np.size(np.unique(y))
    # n_init为k-means算法迭代次数
    kmeans_assignments, _ = get_cluster_sols(X_rep, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10, 'random_state': 42}) 
    if np.min(y) == 1:
        y = y - 1
    return clustering_metric(y, kmeans_assignments)
          
 