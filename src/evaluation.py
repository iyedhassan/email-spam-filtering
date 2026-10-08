def accuracy_from_scratch(y_true, y_pred):
    correct = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    return correct / len(y_true)


def confusion_matrix_from_scratch(y_true, y_pred):
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    for true, pred in zip(y_true, y_pred):

        if true == 0 and pred == 0:
            true_negative += 1

        elif true == 0 and pred == 1:
            false_positive += 1

        elif true == 1 and pred == 0:
            false_negative += 1

        elif true == 1 and pred == 1:
            true_positive += 1

    return [
        [true_negative, false_positive],
        [false_negative, true_positive]
    ]


def precision_from_scratch(y_true, y_pred):
    true_positive = 0
    false_positive = 0

    for true, pred in zip(y_true, y_pred):

        if true == 1 and pred == 1:
            true_positive += 1

        elif true == 0 and pred == 1:
            false_positive += 1

    if true_positive + false_positive == 0:
        return 0

    return true_positive / (true_positive + false_positive)


def recall_from_scratch(y_true, y_pred):
    true_positive = 0
    false_negative = 0

    for true, pred in zip(y_true, y_pred):

        if true == 1 and pred == 0:
            false_negative += 1

        elif true == 1 and pred == 1:
            true_positive += 1

    if true_positive + false_negative == 0:
        return 0

    return true_positive / (true_positive + false_negative)


def f1_score_from_scratch(y_true, y_pred):
    precision = precision_from_scratch(y_true, y_pred)
    recall = recall_from_scratch(y_true, y_pred)

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)