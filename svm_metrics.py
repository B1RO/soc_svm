import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def accuracy_score(y_true, y_predict):
    if len(y_predict) != len(y_true):
        raise ValueError("The length of the list of predicted labels does not match the length of the list of true "
                         "labels")
    return sum(y_predict == y_true) / len(y_true)


def balanced_accuracy_score(y_true, y_predict):
    label_indexes = lambda v: [i for i, x in enumerate(y_true) if x == v]
    accuracy_scores = [accuracy_score(y_true[label_indexes(label)], y_predict[label_indexes(label)]) for label in
                       set(y_true)]
    return sum(accuracy_scores) / len(set(y_true))


def true_positives(y_true, y_predict):
    true_positive_indexes = [i for i, x in enumerate(y_true) if x == 1]
    return sum(y_true[true_positive_indexes] == y_predict[true_positive_indexes])


def false_positives(y_true, y_predict):
    predicted_positive_indxes = [i for i, x in enumerate(y_predict) if x == 1]
    return sum(y_true[predicted_positive_indxes] != y_predict[predicted_positive_indxes])


def true_negatives(y_true, y_predict):
    true_negative_indexes = [i for i, x in enumerate(y_true) if x == -1]
    return sum(y_true[true_negative_indexes] == y_predict[true_negative_indexes])


def false_negatives(y_true, y_predict):
    predicted_negative_indexes = [i for i, x in enumerate(y_predict) if x == -1]
    return sum(y_true[predicted_negative_indexes] != y_predict[predicted_negative_indexes])


def precision_score(y_true, y_predict):
    tp = true_positives(y_true, y_predict)
    fp = false_positives(y_true, y_predict)
    return tp / (tp + fp)


def recall_score(y_true, y_predict):
    tp = true_positives(y_true, y_predict)
    fn = false_negatives(y_true, y_predict)
    return tp / (tp + fn)


def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    return 2 * (precision * recall) / (precision + recall)


def specificity_score(y_true, y_predict):
    tn = true_negatives(y_true, y_predict)
    fp = false_positives(y_true, y_predict)
    return tn / (tn + fp)


def AOC_score(y_true, y_predict):
    tpr = recall_score(y_true, y_predict)
    fpr = 1 - specificity_score(y_true, y_predict)
    return np.trapz([0, tpr, 1, 0], [0, fpr, 1, 1])


def plot_auc(y_true, y_raw):
    tpr = np.empty(101)
    fpr = np.empty(101)
    tpr[100] = 0;
    fpr[100] = 1;
    foo = np.linspace(min(y_raw), max(y_raw), 100)

    for i, treshold in enumerate(foo):
        y_predict = np.sign(y_raw - treshold)
        tpr[i] = recall_score(y_true, y_predict)
        fpr[i] = 1 - specificity_score(y_true, y_predict)

    plt.plot(fpr, tpr, 'p')
    plt.title('Receiver Operating Characteristic')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1:])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in zip(ft[: -1], ft[1:]):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area
