from GaussianNaiveBayesClassifier import Classifier
from Evaluation import Evaluation
from ErrorAnalysis import ErrorAnalysis


def main():
    nb = Classifier(2, 3, 0.05)
    nb.train("training.txt")
    nb.test("test.txt")
    Evaluation(nb)
    ErrorAnalysis(nb)


main()
