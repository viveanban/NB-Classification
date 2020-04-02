# TODO: Remember to consider the size of V (e.g V=0, size_v = 26, V=1, size_v1 = 26*2, V=2, size_v2 = isalpha + 26*2)
from nltk import ngrams
import math

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

    def train(self, training_file):

        global ngram_frequency_per_language
        global language_frequency
        global total_nb_of_tweets

        # open file, get content
        file = open(training_file, encoding='utf-8')
        for line in file:
            # Verify line is not empty
            if line.rstrip().__len__() == 0:
                continue

            total_nb_of_tweets += 1

            # Split line into words
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
                ngram_frequency_per_language[language][ngram] = dict(ngram_frequency_per_language[language]).get(ngram, 0) + 1

        file.close()

    def is_in_vocab(self, char: str) -> bool:
        if self.vocab == 0:
            return 97 <= ord(char) <= 122
        elif self.vocab == 1:
            return 97 <= ord(char) <= 122 or 65 <= ord(char) <= 90
        elif self.vocab == 2:
            return str(char).isalpha()

    def test(self, test_file):
        self.sum_of_freq()
        # open file, get content
        file = open(test_file, encoding='utf-8')
        for line in file:
            ## Verify line is not empty
            if line.rstrip().__len__() == 0:
                continue

            ## Split line into words
            words = line.split()

            id = words[0]
            correct_language = words[2]
            ngram_list = self.get_ngrams_given_word_list(words[3:])

            for language in ngram_frequency_per_language:
                nb_score = self.calculate_nb_score(words[3:], language)

        file.close()

    def get_most_likely_language(self, sentence: str):
        result_map = dict()
        for language in ngram_frequency_per_language:
            result_map[language] = self.calculate_nb_score(sentence, language)

        # some logic to return highest

    def calculate_nb_score(self, tweet: str, lang: str):
        global total_nb_of_tweets
        global language_frequency

        nGramFrequencies = ngram_frequency_per_language.get(lang)

        prior_prob = math.log(self.prior_probability(lang), 10)
        ngram_list = self.get_ngrams_given_word_list(tweet)

        conditional_prob = 0
        for ngram in ngram_list:
            conditional_prob += math.log(self.getConditionalProbability(list(ngram), lang), 10)

        return prior_prob + conditional_prob

    def prior_probability(self, lang:str):
        global total_nb_of_tweets
        return language_frequency[lang] / total_nb_of_tweets

    def get_ngrams_given_word_list(self, tweet_list: list):
        ngram_list = list()

        for word in tweet_list:
            if self.vocab == 0:
                word = word.lower()

            for index in range(0, len(word)):
                char = word[index]
                ngram = char

                if not self.is_in_vocab(char):
                    continue

                next_index = index + 1
                while next_index <= len(word) - 1 and self.is_in_vocab(word[next_index]) and len(ngram) < self.nGram_size:
                    char = word[next_index]
                    ngram += char
                    next_index += 1

                if len(ngram) == self.nGram_size:
                    ngram_list.append(ngram)

        return ngram_list

    def getConditionalProbability(self, ngram, lang):
        # P(ngram| lang) = #ngram/total frequency in language
        frequency = ngram_frequency_per_language.get(lang).get(ngram, 0)
        total = total_ngram_freq_in_lang[lang]
        return (frequency + self.smoothing_value) / (
                total + self.total_ngrams_possible_in_vocab() * self.smoothing_value)

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
