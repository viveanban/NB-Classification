from GaussianNaiveBayesClassifier import Classifier
from Evaluation import Evaluation

def main():
    nb = Classifier(0, 1, 0.25)
    nb.train("training.txt")
    nb.test("test.txt")
    Evaluation(nb)

main()