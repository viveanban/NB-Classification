from GaussianNaiveBayesClassifier import Classifier
from Evaluation import Evaluation
from ErrorAnalysis import ErrorAnalysis
from CustomizedModel import CustomizedModel
import sys


def main():
    # # # Read from standard input
    # vocab = int(sys.argv[1])
    # n = int(sys.argv[2])
    # smoothing_value = float(sys.argv[3])
    # training_file = sys.argv[4]
    # test_file = sys.argv[5]

    nb = Classifier(2, 2, 0.3)
    # nb = CustomizedModel(vocab, n, smoothing_value)
    nb.train('training.txt')
    nb.test('test.txt')
    Evaluation(nb)
    ErrorAnalysis(nb)

main()
