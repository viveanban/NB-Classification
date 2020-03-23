from Classifier import Classifier


def main():
    nb =  Classifier(0, 1, 0.5, "res/training.txt", "res/test.txr")
    nb.train()
main()