from pathlib import Path

nb_correct_guesses = 0
nb_total_guesses = 0

languages = {"es", "eu", "fr", "gl", "en", "pt"}
true_pos = dict()
false_pos = dict()
false_neg = dict()
f1_per_language = dict()
precision_per_language = dict()
recall_per_language = dict()

class Evaluation:

    def __init__(self, trace_file: str, vocab, nGram_size, smoothing_value):
        Path("Outputs").mkdir(parents=True, exist_ok=True)

        file_name = f'Outputs/eval_{vocab}_{nGram_size}_{smoothing_value}.txt'
        self.eval_file = open(file_name, "w+")
        self.read_trace_file(trace_file)
        self.compute_accuracy(nb_correct_guesses, nb_total_guesses)
        self.compute_precision()
        self.compute_recall()
        self.compute_per_class_F1()
        self.compute_macro_f1()
        self.compute_weighted_f_1()
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
            guessed_language = line_elements[1]
            correct_language = line_elements[3]
            isCorrect = True if line_elements[4].__eq__("correct") else False

            if isCorrect:
                true_pos[correct_language] = true_pos.get(line_elements[1], 0) + 1
                nb_correct_guesses += 1
            else:
                false_neg[correct_language] = false_neg.get(correct_language, 0) + 1
                false_pos[guessed_language] = false_pos.get(guessed_language, 0) + 1

        file.close()

    def compute_accuracy(self, correct_guesses, total_guesses):
        # % of instances of the test algo correctly classifies
        accuracy = correct_guesses/total_guesses
        self.eval_file.write(f'{accuracy}\n')

    def compute_precision(self):
        result = ""
        for lang in languages:
            precision = self.get_precision_for_language(lang)
            precision_per_language[lang] = precision
            result += f'{precision}-{lang} '

        result = result.rstrip()
        self.eval_file.write(result + "\n")

    def get_precision_for_language(self, lang):
        return 0 if true_pos.get(lang, 0) == 0 else true_pos.get(lang, 0) / (true_pos.get(lang, 0) + false_pos.get(lang,0))

    def compute_recall(self):
        result = ""
        for lang in languages:
            recall = self.get_recall_for_language(lang)
            recall_per_language[lang] = recall
            result += f'{recall}-{lang}  '

        result = result.rstrip()
        self.eval_file.write(result + "\n")

    def get_recall_for_language(self, lang):
        return 0 if true_pos.get(lang, 0) == 0 else true_pos.get(lang, 0) / (true_pos.get(lang, 0) + false_neg.get(lang, 0))

    def compute_per_class_F1(self):
        result = ""
        for lang in languages:
            f1 = self.get_f_1_for_language(lang)
            f1_per_language[lang] = f1
            result += f'{f1}-{lang}  '

        result = result.rstrip()
        self.eval_file.write(result + "\n")

    def get_f_1_for_language(self, lang):
        recall = recall_per_language.get(lang, 0)
        precision = precision_per_language.get(lang, 0)
        f1 = 0 if (precision + recall is 0) else (2 * precision * recall) / (precision + recall)
        return f1

    def compute_macro_f1(self):
        result = ""
        macro_f1 = 0
        for lang in languages:
            macro_f1 += f1_per_language.get(lang, 0)

        macro_f1 = macro_f1/len(languages)
        result += f'{macro_f1}'
        self.eval_file.write(result + "  ")

    def compute_weighted_f_1(self):
        result = ""
        weighted_f1 = 0
        for lang in languages:
            weighted_f1 += (true_pos.get(lang, 0)+false_neg.get(lang, 0))*(f1_per_language.get(lang, 0))

        weighted_f1 = weighted_f1 / nb_total_guesses
        result += f'{weighted_f1}'
        self.eval_file.write(result + "\n")

    def computer_f1_measure(self):
        pass

    def compute_macro_f1_measure(self):
        pass

    def compute_weighted_avg_f1_measure(self):
        pass
