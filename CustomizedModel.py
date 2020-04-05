from GaussianNaiveBayesClassifier import Classifier
from nltk.corpus import stopwords

class CustomizedModel(Classifier):

    def __init__(self, vocab, nGram_size, smoothing_value):
        super().__init__(vocab, nGram_size, smoothing_value)


    def calculate_nb_score(self, tweet_list: list, lang: str):

        prior_prob = self.prior_probability(lang)
        ngram_list = self.get_ngrams_given_word_list(tweet_list)
        self.languageMap  = {"en":"english", "es":"spanish", "ca":"catalan", "gl":"galician", "pt":"portuguese", "eu":"basque"}

        conditional_prob = 0
        for ngram in ngram_list:
            conditional_prob += self.conditional_probability(ngram, lang)

        return self.get_frequency_of_clues(lang, tweet_list) * (prior_prob + conditional_prob)

    def get_frequency_of_clues(self, lang, tweet_list):

        score = 0
        stopWords = set(stopwords.words(self.languageMap[lang]))
        for word in tweet_list:
            if word in stopWords:
                score += 1
        return score