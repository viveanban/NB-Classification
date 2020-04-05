from GaussianNaiveBayesClassifier import Classifier


class CustomizedModel(Classifier):

    def __init__(self, vocab, nGram_size, smoothing_value):
        super().__init__(vocab, nGram_size, smoothing_value)
        self.cluesDb = dict()
        self.cluesDb["es"] = {"é", "á", "ó", "ú", "ü", "í", "ü", "ñ"}
        self.cluesDb["fr"] = {"ç", "à", "è", "î", "û", "ô", "ö", "ï", "ü", "ù", "`"}
        self.cluesDb["ca"] = {"ç", "á", "é", "í", "ó", "ú", "à", "è", "î", "û", "ô", "ö", "ï", "ü", "ù", "`", "õ", "ñ"}

    def calculate_nb_score(self, tweet_list: list, lang: str):

        prior_prob = self.prior_probability(lang)
        ngram_list = self.get_ngrams_given_word_list(tweet_list)

        conditional_prob = 0
        for ngram in ngram_list:
            conditional_prob += self.conditional_probability(ngram, lang)

        return self.get_frequency_of_clues(lang, tweet_list) * 0.2 * (prior_prob + conditional_prob)

    def get_frequency_of_clues(self, lang, tweet_list):
        count = self.get_sentence_character_frequency(tweet_list)
        frequency_of_clues = 0

        for lang in self.cluesDb:
            for clue in self.cluesDb[lang]:
                frequency_of_clues += count.get(clue, 0)

        return frequency_of_clues

    def get_sentence_character_frequency(self, tweet_list: list):
        count ={}
        for tweet_word in tweet_list:
            list_of_characters = tweet_word.split()
            for character in list_of_characters:
                if character in count:  # check if it exists in dictionary
                    count[tweet_word] += 1
                else:
                    count[tweet_word] = 1  # first occurrence of character
        return count