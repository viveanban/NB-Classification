import GaussianNaiveBayesClassifier


class CustomizedModel(GaussianNaiveBayesClassifier):

    def __init__(self, original_text):
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

        return prior_prob + conditional_prob

    def frequency_of_words(self, lang, sentence):
        count = {}
        sentence = sentence.split()

        for i in sentence:
            if i in count:  # check if it exists in dictionary
                count[i] += 1
            else:
                count[i] = 1  # first occurrence of character

        frequency = 0

        for clue in self.cluesDb[lang]:
            frequency += count[clue]

        return frequency
