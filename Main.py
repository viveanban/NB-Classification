from GaussianNaiveBayesClassifier import Classifier

def main():
    nb = Classifier(2, 2, 0)
    nb.train("training.txt")
    nb.test("test.txt")
    print("debug")

main()

