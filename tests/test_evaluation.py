from src.evaluation import (
    accuracy_from_scratch,
    precision_from_scratch,
    recall_from_scratch,
    f1_score_from_scratch,
    confusion_matrix_from_scratch
)


def test_accuracy():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    result = accuracy_from_scratch(y_true, y_pred)

    assert result == 1.0


def test_precision():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = precision_from_scratch(y_true, y_pred)

    assert result == 2 / 3


def test_recall():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = recall_from_scratch(y_true, y_pred)

    assert result == 1.0


def test_f1_score():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = f1_score_from_scratch(y_true, y_pred)

    expected = 2 * ((2 / 3) * 1.0) / ((2 / 3) + 1.0)

    assert result == expected


def test_confusion_matrix():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = confusion_matrix_from_scratch(y_true, y_pred)

    assert result == [
        [1, 1],
        [0, 2]
    ]