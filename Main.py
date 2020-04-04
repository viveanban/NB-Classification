from GaussianNaiveBayesClassifier import Classifier
from Evaluation import Evaluation

def main():
    nb = Classifier(3, 2, 0.05)
    nb.train("training.txt")
    nb.test("test.txt")
    Evaluation(nb)

main()