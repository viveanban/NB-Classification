from pathlib import Path

from GaussianNaiveBayesClassifier import Classifier

frequency_of_guesses_per_language = dict()
confusion_matrix = dict()


class ErrorAnalysis:

    def __init__(self, nb: Classifier):
        print("Error Analysis started")
        Path("Outputs").mkdir(parents=True, exist_ok=True)

        file_name = f'Outputs/EM{nb.vocab}_{nb.nGram_size}_{nb.smoothing_value}.txt'
        trace_file = nb.trace_file.name
        self.error_file = open(file_name, "w+")
        self.read_trace_file(trace_file)
        self.nb = nb

    def read_trace_file(self, trace_file: str):

        file = open(trace_file, encoding='utf-8')
        for line in file:
            if line.rstrip().__len__() == 0:
                continue

            line_elements = line.split()
            guessed_language = line_elements[1]
            correct_language = line_elements[3]

            # create confusion matrix
            frequency_of_guesses_per_language[guessed_language] = frequency_of_guesses_per_language.get(
                guessed_language, 0) + 1
            confusion_matrix[guessed_language] = confusion_matrix.get(guessed_language, dict())
            confusion_matrix[guessed_language][correct_language] = dict(confusion_matrix.get(guessed_language)).get(
                correct_language, 0) + 1

        for guess in confusion_matrix:
            for correct in confusion_matrix[guess]:
                confusion_matrix[guess][correct] = confusion_matrix[guess][correct] / frequency_of_guesses_per_language[
                    guess]

        self.output_confusion_analysis()
        file.close()

    def output_confusion_analysis(self):
        for guess in confusion_matrix:
            self.error_file.write(guess + ":  ")
            for correct in confusion_matrix[guess]:
                self.error_file.write(str(confusion_matrix[guess][correct]) + "-" + correct + "  ")
            self.error_file.write("\n")
