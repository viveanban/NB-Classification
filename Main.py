from GaussianNaiveBayesClassifier import Classifier

def main():
    nb = Classifier(2, 2, 0)
    nb.train("practice")
    nb.test("practice")
    print("debug")

main()