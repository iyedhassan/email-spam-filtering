from src.model import NaiveBayesSpamClassifier


def test_model_training():

    X = [
        "hello how are you",
        "call me tonight",
        "free money now",
        "win a free prize"
    ]

    y = [
        0,
        0,
        1,
        1
    ]

    model = NaiveBayesSpamClassifier()

    model.fit(X, y)

    assert 0 in model.classes
    assert 1 in model.classes


def test_prediction():

    X = [
        "hello how are you",
        "call me tonight",
        "free money now",
        "win a free prize"
    ]

    y = [
        0,
        0,
        1,
        1
    ]

    model = NaiveBayesSpamClassifier()

    model.fit(X, y)

    prediction = model.predict_one("free money")

    assert prediction in [0, 1]