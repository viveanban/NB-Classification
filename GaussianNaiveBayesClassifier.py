import math
from pathlib import Path
import string

# Variable Declaration
ngram_frequency_per_language = dict()
language_frequency = dict()
total_nb_of_tweets = 0
total_ngram_freq_in_lang = dict()


class Classifier:

    def __init__(self, vocab, nGram_size, smoothing_value):
        self.vocab = vocab
        self.nGram_size = nGram_size
        self.smoothing_value = smoothing_value

        # output file setup
        Path("Outputs").mkdir(parents=True, exist_ok=True)
        trace_file_name = f'Outputs/trace_{self.vocab}_{self.nGram_size}_{self.smoothing_value}.txt'
        self.trace_file = open(trace_file_name, "w+", encoding='utf-8')

    def train(self, training_file):

        print("Training started")

        global ngram_frequency_per_language
        global language_frequency
        global total_nb_of_tweets

        # open file, get content
        file = open(training_file, encoding='utf-8')
        for line in file:

            # check if empty line
            if line.rstrip().__len__() == 0:
                continue

            total_nb_of_tweets += 1

            words = line.split()

            # Add language to map of all languages
            language = words[2]

            language_model = ngram_frequency_per_language.get(language, 0)

            if language_model == 0:
                ngram_frequency_per_language[language] = dict()

            # Update frequency table
            language_frequency[language] = language_frequency.get(language, 0) + 1

            # Extract the ngrams from the tweet
            ngram_list = self.get_ngrams_given_word_list(words[3:])

            # Update the language model
            for ngram in ngram_list:
                ngram_frequency_per_language[language][ngram] = dict(ngram_frequency_per_language[language]).get(ngram,
                                                                                                                 0) + 1

        file.close()
        print("Training complete")

    def is_in_vocab(self, char: str) -> bool:
        if self.vocab == 0:
            return 97 <= ord(char) <= 122
        elif self.vocab == 1:
            return 97 <= ord(char) <= 122 or 65 <= ord(char) <= 90
        elif self.vocab == 2:
            return str(char).isalpha()
        elif self.vocab == 3:
            return 97 <= ord(char) <= 122 or string.punctuation.__contains__(char)
        elif self.vocab == 4:
            return 97 <= ord(char) <= 122 or str(char).isalpha() or string.punctuation.__contains__(char)

    def test(self, test_file):
        print("Testing started")
        self.sum_of_freq()
        # open file, get content
        file = open(test_file, encoding='utf-8')
        for line in file:
            if line.rstrip().__len__() == 0:
                continue

            words = line.split()

            id = words[0]
            correct_language = words[2]
            all_scores = self.get_all_nb_scores_for_tweet(words[3:])

            guessed_language = max(all_scores, key=all_scores.get)
            score = all_scores[guessed_language]

            self.trace_output(id, guessed_language, score, correct_language)

        file.close()
        self.trace_file.close()
        print("Testing complete")

    def get_all_nb_scores_for_tweet(self, tweet_list: str):
        all_scores = dict()

        for language in ngram_frequency_per_language:
            nb_score = self.calculate_nb_score(tweet_list, language)
            all_scores[language] = nb_score

        return all_scores

    def calculate_nb_score(self, tweet_list: list, lang: str):
        global total_nb_of_tweets
        global language_frequency

        prior_prob = self.prior_probability(lang)
        ngram_list = self.get_ngrams_given_word_list(tweet_list)

        conditional_prob = 0
        for ngram in ngram_list:
            conditional_prob += self.conditional_probability(ngram, lang)

        return prior_prob + conditional_prob

    def prior_probability(self, lang: str):
        global total_nb_of_tweets
        prob = language_frequency[lang] / total_nb_of_tweets
        return math.log(prob, 10) if prob > 0 else -math.inf

    def get_ngrams_given_word_list(self, tweet_list: list):
        ngram_list = list()

        for word in tweet_list:

            if self.vocab == 0 or self.vocab == 3 or self.vocab == 4:
                word = word.lower()

            for index in range(0, len(word)):

                char = word[index]
                ngram = char

                if not self.is_in_vocab(char):
                    continue

                next_index = index + 1
                while next_index <= len(word) - 1 and self.is_in_vocab(word[next_index]) and len(
                        ngram) < self.nGram_size:
                    char = word[next_index]
                    ngram += char
                    next_index += 1

                if len(ngram) == self.nGram_size:
                    ngram_list.append(ngram)

        return ngram_list

    def conditional_probability(self, ngram, lang):
        # P(ngram| lang) = #ngram/total frequency in language

        frequency = ngram_frequency_per_language.get(lang).get(ngram, 0)
        total = total_ngram_freq_in_lang[lang]
        prob = (frequency + self.smoothing_value) / (
                total + self.total_ngrams_possible_in_vocab() * self.smoothing_value)
        return math.log(prob, 10) if prob > 0 else -math.inf

    def sum_of_freq(self):

        global total_ngram_freq_in_lang

        for lang in ngram_frequency_per_language:
            language_ngram_map = ngram_frequency_per_language[lang]
            total_occurence = 0
            for ngram in language_ngram_map:
                total_occurence += language_ngram_map[ngram]

            total_ngram_freq_in_lang[lang] = total_occurence

    def total_ngrams_possible_in_vocab(self):
        if self.vocab == 0:
            return math.pow(26, self.nGram_size)
        elif self.vocab == 1:
            return math.pow(26 * 2, self.nGram_size)
        elif self.vocab == 2:
            return math.pow(116766, self.nGram_size)
        elif self.vocab == 3:
            return math.pow(26 + 32, self.nGram_size)
        elif self.vocab == 4:
            return math.pow(26 + 32 + 116766, self.nGram_size)

    def trace_output(self, id, guessed_label, score, correct_label):
        annotation = "correct" if guessed_label.__eq__(correct_label) else "wrong"
        # TODO: Make sure to print the score in scientific notation
        score = "{:.2E}".format(score)
        result = f'{id}  {guessed_label}  {score}  {correct_label}  {annotation}\n'
        self.trace_file.write(result)

    def eval_output(self):
        pass
