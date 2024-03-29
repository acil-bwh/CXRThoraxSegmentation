import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import numpy as np
from sklearn.metrics import roc_auc_score


# METRICS
def younden_idx(real, pred):
    fpr, tpr, thresholds = metrics.roc_curve(real, pred)
    return thresholds[np.argmax(tpr-fpr)]


def pred_recall_thres(precision, recall, thresholds):
    pr_max = thresholds[np.argmax(precision+recall)]
    try:
        pr_cut= thresholds[np.where(precision == recall)][0]
    except:
        pr_cut= thresholds[np.where(precision == recall)]
    if not isinstance(pr_cut, float):
        pr_cut = 0
    return pr_max, pr_cut


def f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)


def AUC_CI(y_true, y_pred, n_bootstraps, rng_seed = 42):
    aucs = []
    rng = np.random.RandomState(rng_seed)
    for _ in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        aucs.append(score)
    return aucs, (np.percentile(aucs,2.5), np.percentile(aucs,97.5))


# PLOTS
def AUC_plot(fpr, tpr, thresholds, auc, title = '', aucci = None):
    fig, ax = plt.subplots()
    i = np.argmax(tpr-fpr)
    th = thresholds[i]
    x = fpr[i]
    y = tpr[i]
    if aucci != None:
        ax.plot(fpr,tpr, "g-", label= "AUC=" + str(round(auc, 2)) + 
                                        " (" + str(round(aucci[0],2)) +
                                        "-" + str(round(aucci[1],2)) +
                                        ")")
    else:
        ax.plot(fpr,tpr, "g-", label="AUC="+str(round(auc, 2)))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    try:
        ax.plot([x, x], [0, y], "r:")
        ax.plot([0, x], [y, y], "r:")
    except:
        print('plot except')
    ax.plot([x], [y], "ro", label="th="+str(round(th,2))) 
    ax.legend(loc=4)
    ax.set_title(title)
    return fig


def pred_recall_plot(precision, recall, thresholds):
    fig, ax = plt.subplots()
    i = np.argmax(precision+recall)
    x = recall[i]
    y = precision[i]
    th = thresholds[i]
    ax.plot(recall, precision, "g-")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    try:
        ax.plot([x, x], [0, y], "r:")
        ax.plot([0, x], [y, y], "r:")
    except:
        print('plot except')
    ax.plot([x], [y], "ro", label="th="+str(round(th,2))) 
    ax.legend(loc=4)
    ax.set_title('Precision-Recall curve')
    return fig


def plot_precision_recall_vs_threshold(precision, recall, thresholds):
    fig, ax = plt.subplots()
    f1 = f1_score(precision, recall)
    x_f = thresholds[np.argmax(f1)]
    y_f = f1[np.argmax(f1)]
    x = thresholds[np.where(precision == recall)]
    y = precision[np.where(precision == recall)]
    ax.axis([0 ,1, np.min(precision), 1])
    ax.plot(thresholds, precision[:-1], "r-", label="Precision", linewidth=2)
    ax.plot(thresholds, recall[:-1], "b-", label="Recall", linewidth=2)
    ax.plot(thresholds, f1[:-1], "m--", label="F1 score", linewidth=2)
    try:
        ax.plot([x], [y], "bo", label="th="+str(round(x[0],2)))
    except:
        print('th exception')
    try:
        ax.plot([x, x], [0, y], "k:")
    except:
        print('plot except')
    ax.plot([x_f, x_f], [0, y_f], "k:")
    ax.plot([x_f], [y_f], "ro", label="th="+str(round(x_f,2))) 
    ax.set_xlabel("Threshold")
    ax.grid(True)
    ax.legend()
    return fig


def save_plot(plot, folder, title):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plot.savefig(os.path.join(folder, title + '.png'))


# DICCIONARIOS
def extract_max(array):
    for i in range(array.shape[0]):
        max = np.argmax(array[i,:])
        array[i,:] = 0
        array[i,max] = 1
    return array


# Cada una de las clases genera un diccionario con las metricas y los plots
def metrics_per_class(name, real, pred):
    metricas = {}
    fpr, tpr, auc_thresholds = metrics.roc_curve(real, pred)
    auc = metrics.auc(fpr, tpr)
    metricas['auc_' + name] = auc
    metricas['younden_'+ name] = younden_idx(real, pred)
    precision, recall, pr_thresholds = metrics.precision_recall_curve(real, pred)
    metricas['pr_max_'+ name], metricas['pr_cut_'+ name] = pred_recall_thres(precision, 
                                                                            recall, 
                                                                            pr_thresholds)
    print(f'metricas clase {name} calculadas')
    plots = {}
    plots['pred_rec_plot_' + name] = pred_recall_plot(precision, recall, pr_thresholds)
    plots['auc_plot_' + name] = AUC_plot(fpr, tpr, auc_thresholds, auc)
    plots['pr_re_th_plot_' + name] = plot_precision_recall_vs_threshold(precision, 
                                                                        recall, 
                                                                        pr_thresholds)
    print(f'plots clase {name} realizados')
    return metricas, plots


def binarize(array, threshold):
    array[array >= threshold] = 1
    array[array < threshold] = 0
    return array


def metricas_dict(real, pred):
    metrics_dict = {}
    plot_dict = {}
    metricas, plots = metrics_per_class('', real, pred[:,0])
    metrics_dict.update(metricas)
    plot_dict.update(plots)
    thresholds = ['younden_','pr_max_','pr_cut_']
    for threshold in thresholds:
        binar = binarize(pred[:,0].copy(), metricas[threshold])
        metrics_dict['f1_score_' + threshold] = metrics.f1_score(real, binar, 
                                                                average = 'weighted')
        metrics_dict['precision_score_' + threshold] = metrics.precision_score(real, 
                                                                            binar, 
                                                                            average = 'weighted')
        metrics_dict['recall_score_' + threshold] = metrics.recall_score(real, 
                                                                        binar, 
                                                                        average = 'weighted')
        metrics_dict['accuracy_score_' + threshold] = metrics.accuracy_score(real, binar)

    binar = binarize(pred[:,0].copy(), 0.5)
    metrics_dict['f1_score_' + str(0.5)] = metrics.f1_score(real, binar, 
                                                            average = 'weighted')
    metrics_dict['precision_score_' + str(0.5)] = metrics.precision_score(real, 
                                                                        binar, 
                                                                        average = 'weighted')
    metrics_dict['recall_score_' + str(0.5)] = metrics.recall_score(real, 
                                                                    binar, 
                                                                    average = 'weighted')
    metrics_dict['accuracy_score_' + str(0.5)] = metrics.accuracy_score(real, binar)
    
    binar = extract_max(pred.copy())[:,0]
    metrics_dict['f1_score_' + 'max'] = metrics.f1_score(real, binar, 
                                                            average = 'weighted')
    metrics_dict['precision_score_' + 'max'] = metrics.precision_score(real, 
                                                                        binar, 
                                                                        average = 'weighted')
    metrics_dict['recall_score_' + 'max'] = metrics.recall_score(real, 
                                                                    binar, 
                                                                    average = 'weighted')
    metrics_dict['accuracy_score_' + 'max'] = metrics.accuracy_score(real, binar)

    return metrics_dict, plot_dict

