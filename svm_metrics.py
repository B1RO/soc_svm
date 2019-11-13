def accuracy_score(y_true, y_predidct):
    if len(y_predidct) != len(y_true):
        raise ValueError("The length of the list of predicted labels does not match the length of the list of true "
                         "labels")
    return sum(y_predidct == y_true) / len(y_true)


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

def f1_score(y_true,y_predict):
    precision = precision_score(y_true,y_predict)
    recall = recall_score(y_true,y_predict)
    return 2 * (precision * recall) / (precision + recall)

def specificity_score(y_true,y_predict):
    tn = true_negatives(y_true, y_predict)
    fp = false_positives(y_true, y_predict)
    return tn/(tn+fp)

