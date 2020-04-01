nGram_frequency_per_language = dict()
language_frequency = dict()
total_nb_of_tweets = 0
#TODO: Remember to consider the size of V (e.g V=0, size_v = 26, V=1, size_v1 = 26*2, V=2, size_v2 = isalpha + 26*2)

class Classifier:
    def __init__(self, vocab, nGram_size, smoothing_value):
        self.vocab = vocab
        self.nGram_size = nGram_size
        self.smoothing_value = smoothing_value

    def train(self, training_file):
        global nGram_frequency_per_language
        global language_frequency
        global total_nb_of_tweets

        # open file, get content
        file = open(training_file, encoding='utf-8')
        for line in file:
            ## Verify line is not empty
            if line.rstrip().__len__() == 0:
                continue

            total_nb_of_tweets +=  1

            ## Split line into words
            words = line.split()

            ## Add language to map of all languages
            language = words[2]
            languageModel = nGram_frequency_per_language.get(language, 0)

            if languageModel == 0:
                nGram_frequency_per_language[language] = dict()

            ## Update frequency table
            language_frequency[language] = language_frequency.get(language, 0) + 1

            for tweetWord in words[3:]:
                if self.vocab == 0:
                    tweetWord = tweetWord.lower()

                listOfChar = list(tweetWord)
                for index in range(0, listOfChar.__len__()):
                    char: str = listOfChar[index]

                    if self.is_in_vocab(self.vocab, char):
                        nGram: str = char
                        if self.nGram_size == 2:
                            if not ((index + 1) < listOfChar.__len__() and self.is_in_vocab(self.vocab, listOfChar[index + 1])):
                                continue
                            nGram += str(listOfChar[index + 1])
                        elif self.nGram_size == 3:
                            if not ((index + 2) < listOfChar.__len__() and self.is_in_vocab(self.vocab, listOfChar[
                                index + 1]) and self.is_in_vocab(self.vocab, listOfChar[index + 2])):
                                continue
                            nGram += str(listOfChar[index + 1]) + str(listOfChar[index + 2])

                        nGram_frequency_per_language.get(language)[nGram] = dict(nGram_frequency_per_language[language]).get(nGram, 0) + 1
        file.close()

    def is_in_vocab(self, vocab: int, char: str) -> bool:
        if vocab == 0:
            return 97 <= ord(char) <= 122
        elif vocab == 1:
            return 97 <= ord(char) <= 122 or 65 <= ord(char) <= 90
        elif vocab == 2:
            return str(char).isalpha()

    def test(self, test_file):
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

            for tweetWord in words[3:]:
                self.score(tweetWord)
        file.close()

    def score(self, tweet:str):
        # TODO: Don't forget smoothing
        print()

