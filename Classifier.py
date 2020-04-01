class Classifier:
    def __init__(self, vocab, size, smoothing_value, training_file, testing_file):
        self.vocab = vocab
        self.size = size
        self.smoothing_value = smoothing_value
        self.training_file = training_file
        self.testing_file = testing_file

    def train(self):
        print("Start training!")
        model = dict() ## hashmpa avec les n-gram et leur frequency

        for evidence in self.getTrainingSet():
            print(evidence)

    def getTrainingSet(self):
        input_list = []
        with open(self.training_file) as file:
            for line in file:
                input = line.split("\t", 3)
                print(input)
                if len(input) != 4:
                    raise Exception("Incorrect input format. Verify file: " + self.training_file)
                input_list.append(input)
        return input_list
