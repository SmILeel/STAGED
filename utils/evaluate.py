import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.preprocessing import scale, minmax_scale
from utils.metric import *

def get_best_performance_data(total_err_scores, gt_labels, topk=10, point_adjust=False):
    total_err_scores = total_err_scores.T
    total_features = total_err_scores.shape[0]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    final_meas_list, thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)
    # 0:F1_score, 1:precision_score, 2:recall_score
    th_i = final_meas_list[0].index(max(final_meas_list[0]))
    thresold = thresolds[th_i]
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    if point_adjust:
        adjust_predicts(gt_labels, pred_labels)

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)


    return max(final_meas_list[0]), pre, rec, auc_score, thresold


def get_val_performance_data(total_err_scores,  gt_labels, topk=10, point_adjust=False):
    total_features = total_err_scores.shape[0]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > 1] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    if point_adjust:
        adjust_predicts(gt_labels, pred_labels)

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score


def eval_scores(scores, true_scores, th_steps, return_thresold=True):
    padding = np.zeros(len(true_scores) - len(scores))
    if len(padding) > 0:
        scores = np.concatenate((padding, scores))
    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    precision_meas = [None] * th_steps
    recall_meas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)
        cur_pred = adjust_predicts(true_scores, cur_pred)
        fmeas[i] = f1_score(true_scores, cur_pred)
        precision_meas[i] = precision_score(true_scores, cur_pred)
        recall_meas[i] = recall_score(true_scores, cur_pred)
        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return [fmeas, precision_meas, recall_meas], thresholds
    return [fmeas, precision_meas, recall_meas]


def normalize_error(reconstruct_error, val_error=None, beta=1.5, method='mean_and_std', dataset='swat', clip=False,
                    denoise=True):
    """
    Normalize the test result to eliminate the impact of different dimensions scale and dynamics
    :param reconstruct_error:  T_test * num_dim
    :param val_error: T_val * num_dim
    :param method: the normalize_method
    :param dataset: the uesed dataset
    :param clip: wheather to clip the extreme value in normalized error
    :param denoise: wheather to desert the error between a threshold based on the val_error

    :return: T * num_dim --> the normalized test error
    """

    reconstruct_error = np.array(reconstruct_error)
    # std_list = []
    # for i in range(reconstruct_error.shape[1]):
    #     std_list.append(reconstruct_error[:, i].std())
    if denoise:
        if val_error:
            val_error = np.array(val_error)
            val_error_max = val_error.max(axis=0)
        if dataset == 'swat':
            reconstruct_error[reconstruct_error < beta * val_error_max] = 0
        elif dataset == 'CVACaseStudy':
            reconstruct_error[reconstruct_error < beta * val_error_max] = 0
            # return np.where(reconstruct_error < 0.05, reconstruct_error, 1)
        elif dataset == 'wadi':
            reconstruct_error[reconstruct_error < beta * val_error_max] = 0
        else:
            raise NameError('No such dataset!')
    T, num_dim = reconstruct_error.shape
    epsilon = 1e-3
    for i in range(num_dim):
        if np.max(reconstruct_error[:, i]) == 0:
            continue
        err_median = np.median(reconstruct_error[:, i])
        err_mean = np.mean(reconstruct_error[:, i])
        err_iqr = iqr(reconstruct_error[:, i])
        err_std = np.std(reconstruct_error[:, i])
        # reconstruct_error[:, i] = (reconstruct_error[:, i] - err_median) / ( np.abs(err_iqr) +epsilon)
        if method == 'mean_and_std':
            reconstruct_error[:, i] = np.abs(reconstruct_error[:, i] - err_mean) / (err_std + epsilon)
        elif method == 'std':
            reconstruct_error[:, i] = reconstruct_error[:, i] / (err_std + epsilon)
            # reconstruct_error[:, i] = reconstruct_error[:, i] / (std_list[i] + epsilon)
        elif method == 'median_and_iqr':
            reconstruct_error[:, i] = np.abs(reconstruct_error[:, i] - err_median) / (np.abs(err_iqr) + epsilon)
        elif method == 'median_and_std':
            reconstruct_error[:, i] = np.abs(reconstruct_error[:, i] - err_median) / (np.abs(err_std) + epsilon)
        elif method == 'mean_and_iqr':
            reconstruct_error[:, i] = np.abs(reconstruct_error[:, i] - err_mean) / (np.abs(err_iqr) + epsilon)
        elif method == 'min_max_scale':
            reconstruct_error[:, i] = minmax_scale(reconstruct_error[:, i])
        elif method == 'no_transform':
            pass
        else:
            raise ValueError('No such options!')
    if clip:
        if dataset == 'swat':
            reconstruct_error = reconstruct_error.clip(max=10)
        elif dataset == 'CVACaseStudy':
            reconstruct_error = reconstruct_error.clip(max=12)
        else:
            raise NameError('No such dataset type!')
    return reconstruct_error


def smooth_error(error_score):
    """
    smooth the error using moving average.
    :param error_score: T * num_dim   array
    :return: smoothed error
    """
    before_num = 3
    for i in range(before_num, len(error_score)):
        error_score[i] = np.mean(error_score[i - before_num: i + 1])
    return error_score


def batch_reconstruct_error(batch_data):
    """
    calculate the batch reconstruction error.
    :param batch_data: Graph batch_data
    :return: reconstruct error  [T * num_dim]
    """
    graph_list_test = batch_data.to_data_list()
    for graph in graph_list_test:
        x_raw, x_reconstruct = graph.x, graph.x_
        span_error = []
        for i in range(len(x_raw)):
            feature_error = dtw_reconstruction_error(x_raw, x_reconstruct)
            span_error.append(feature_error)
    return span_error


# Calculate dtw-distance
def dtw_reconstruction_error(x, x_):
    n, m = x.shape[0], x_.shape[0]
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - x_[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n][m]
