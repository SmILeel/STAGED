import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score,classification_report,confusion_matrix, precision_score, recall_score,roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def normalize(seq):
    return (seq - np.min(seq)) / (np.max(seq) - np.min(seq))

def metrics_calculate(labels, scores):
    # scores = anomaly_scoring(values, re_values)

    preds, _ = evaluate1(labels, scores, adj=False)
    preds_, _ = evaluate1(labels, scores, adj=True)

    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    re = recall_score(y_true=labels, y_pred=preds)

    f1_ = f1_score(y_true=labels, y_pred=preds_)
    pre_ = precision_score(y_true=labels, y_pred=preds_)
    re_ = recall_score(y_true=labels, y_pred=preds_)
    auc = roc_auc_score(y_true=labels, y_score=normalize(scores))

    print('F1 score is [%.5f / %.5f] (before adj / after adj), auc score is %.5f.' % (f1, f1_, auc))
    print('Precision score is [%.5f / %.5f], recall score is [%.5f / %.5f].' % (pre, pre_, re, re_))

    return auc, f1, pre, re

def evaluate2(labels, scores, r=0.01, adj = False):
    preds = np.zeros_like(labels)
    idx = scores.argsort()
    ano_num = int(r * len(labels))
    preds[idx[-ano_num:]] = 1
    if adj:
        preds = adjust_predicts(labels, preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    
    return preds, f1


def evaluate1(labels, scores, step=2000, adj=True):
    # best f1
    min_score = min(scores)
    max_score = max(scores)
    best_f1 = 0.0
    best_preds = None
    for th in tqdm(np.linspace(min_score, max_score, step), ncols=70):
        preds = (scores > th).astype(int)
        if adj:
            preds = adjust_predicts(labels, preds)
        f1 = f1_score(y_true=labels, y_pred=preds)
        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds

    return best_preds, best_f1

def evaluate(labels, scores, res_th=None, saveto=None, adj = True):
    '''
    metric for auc/ap
    :param labels:
    :param scores:
    :param res_th:
    :param saveto:
    :return:
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # print(np.median(scores))
    # True/False Positive Rates.
    fpr, tpr, ths = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    # best f1
    best_f1 = 0
    best_threshold = 0
    for threshold in ths:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores > threshold] = 1
        tmp_scores[tmp_scores <= threshold] = 0
        cur_f1 = f1_score(labels, tmp_scores)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold
    

    media = np.median(scores)
    # best_threshold = media
    tmp_scores = scores.copy()
    tmp_scores[scores > media] = 1
    tmp_scores[scores <= media] = 0
    precision = precision_score(labels, tmp_scores)
    recall = recall_score(labels, tmp_scores)
    # best_f1 = f1_score(labels, tmp_scores)

    #threshold f1
    if res_th is not None and saveto is  not None:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= res_th] = 1
        tmp_scores[tmp_scores < res_th] = 0
        print(classification_report(labels,tmp_scores))
        print(confusion_matrix(labels,tmp_scores))
    auc_prc=average_precision_score(labels,scores)
    
    return auc_prc,roc_auc,best_threshold,best_f1, precision, recall


def adjust_predicts(label, pred=None):
    predict = pred.astype(bool)
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(label)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True
    return predict.astype(int)
