from collections import Counter
import math


class NaiveBayesSpamClassifier:

    def __init__(self):
        self.classes = None
        self.vocabulary = set()

        self.class_counts = Counter()
        self.word_counts = {
            0: Counter(),
            1: Counter()
        }

        self.total_words = {
            0: 0,
            1: 0
        }

    def fit(self, X, y):

        self.classes = set(y)

        for message, label in zip(X, y):

            self.class_counts[label] += 1

            words = message.split()

            for word in words:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)
                self.total_words[label] += 1

    def word_probability(self, word, label):

        word_count = self.word_counts[label][word]

        total_words = self.total_words[label]

        vocabulary_size = len(self.vocabulary)

        probability = (
            (word_count + 1)
            / (total_words + vocabulary_size)
        )

        return probability

    def class_probability(self, label):

        total_messages = sum(self.class_counts.values())

        return self.class_counts[label] / total_messages

    def predict_one(self, message):

        words = message.split()

        scores = {}

        for label in self.classes:

            #score = self.class_probability(label)
            score = math.log(self.class_probability(label))

            for word in words:

                probability = self.word_probability(word, label)

                #score *= probability
                score += math.log(probability)

            scores[label] = score

        return max(scores, key=scores.get)

    def predict(self, X):
        predictions = []

        for message in X:
            prediction = self.predict_one(message)
            predictions.append(prediction)

        return predictions