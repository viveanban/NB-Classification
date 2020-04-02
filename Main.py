from GaussianNaiveBayesClassifier import Classifier
from Evaluation import Evaluation

def main():
    nb = Classifier(2, 2, 0)
    nb.train("practice")
    nb.test("practice")
    evaluation = Evaluation("Outputs/trace_2_2_0.txt", 2, 2, 0)
    print("debug")

main()