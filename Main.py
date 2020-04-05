from GaussianNaiveBayesClassifier import Classifier
from Evaluation import Evaluation
from ErrorAnalysis import ErrorAnalysis


def main():
    nb = Classifier(1, 2, 1)
    nb.train("training.txt")
    nb.test("test.txt")
    Evaluation(nb)
    ErrorAnalysis(nb)


main()
