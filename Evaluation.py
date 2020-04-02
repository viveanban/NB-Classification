from pathlib import Path

nb_correct_guesses = 0
nb_total_guesses = 0

correct_guesses_per_language = dict()
total_guesses_per_language = dict()


class Evaluation:

    def __init__(self, trace_file: str, vocab, nGram_size, smoothing_value):
        Path("Outputs").mkdir(parents=True, exist_ok=True)

        file_name = f'Outputs/eval_{vocab}_{nGram_size}_{smoothing_value}.txt'
        self.eval_file = open(file_name, "w+")
        self.read_trace_file(trace_file)
        self.compute_accuracy(nb_correct_guesses, nb_total_guesses)
        self.compute_precision()
        self.eval_file.close()


    def read_trace_file(self, trace_file: str):
        global nb_correct_guesses
        global nb_total_guesses
        # open file, get content
        file = open(trace_file, encoding='utf-8')
        for line in file:
            if line.rstrip().__len__() == 0:
                continue

            # Accuracy
            nb_total_guesses += 1

            line_elements = line.split()

            total_guesses_per_language[line_elements[1]] = total_guesses_per_language.get(line_elements[1], 0) + 1

            if line_elements[4].__eq__("correct"):
                nb_correct_guesses += 1
                correct_guesses_per_language[line_elements[1]] = correct_guesses_per_language.get(line_elements[1], 0) + 1






        file.close()

    def compute_accuracy(self, correct_guesses, total_guesses):
        # % of instances of the test algo correctly classifies
        accuracy = correct_guesses/total_guesses
        self.eval_file.write(f'{accuracy}\n')

    def compute_precision(self):
        result = ""
        for lang in total_guesses_per_language:
            precision = correct_guesses_per_language[lang] / total_guesses_per_language[lang]
            result += f'{precision}  '

        result = result.rstrip()
        self.eval_file.write(result)
        pass

    def compute_recall(self):
        pass

    def computer_f1_measure(self):
        pass

    def compute_macro_f1_measure(self):
        pass

    def compute_weighted_avg_f1_measure(self):
        pass
