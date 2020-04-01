from Classifier import Classifier

# Use hashmap to store frequency for each language
# Use hashmap to store totalite de n-grams dedans for each language
# need a hashmap for each language to keep the count of occurence of each n-gram. (6 hashmaps)
# need to remember total number of tweets
# need number of n-grams per language

languageModels = dict()
frequencyLanguageTable = dict()


def main():
    nb = Classifier(0, 1, 0, "training.txt", "test.txt")
    train(0, 2, "practice")
    print("debug")


## TODO: improvement maybe? should do some checks on the file to ensure it's the correct format?
def train(vocab, nGramSize, training_file):
    # open file, get content
    file = open(training_file, encoding='utf-8')
    for line in file:
        ## Verify line is not empty
        if line.rstrip().__len__() == 0:
            continue

        ## Split line into words
        words = line.split()

        id = words[0]

        ## Add language to map of all languages
        language = words[2]
        languageModel = languageModels.get(language, 0)

        if languageModel == 0:
            languageModels[language] = dict()

        ## Update frequency table
        frequencyLanguageTable[language] = frequencyLanguageTable.get(language, 0) + 1

        for tweetWord in words[3:]:
            listOfChar = list(tweetWord)
            for index in range(0, listOfChar.__len__()):
                char: str = listOfChar[index]
                if vocab == 0:
                    char = char.lower()

                if is_in_vocab(vocab, char):
                    nGram: str = char
                    if nGramSize == 2:
                        if not ((index + 1) < listOfChar.__len__() and is_in_vocab(vocab, listOfChar[index + 1])):
                            break
                        nGram += str(listOfChar[index + 1])
                    elif nGramSize == 3:
                        if not ((index + 2) < listOfChar.__len__() and is_in_vocab(vocab, listOfChar[index + 1]) and is_in_vocab(vocab, listOfChar[index + 2])):
                            break
                        nGram += str(listOfChar[index + 1]) + str(listOfChar[index + 2])


                    languageModels.get(language)[nGram] = dict(languageModels[language]).get(nGram, 0) + 1
    file.close()


def is_in_vocab(vocab: int, char: str) -> bool:
    if vocab == 0:
        return 97 <= ord(char) <= 122
    elif vocab == 1:
        return 97 <= ord(char) <= 122 or 65 <= ord(char) <= 90
    elif vocab == 2:
        return str(char).isalpha()


main()

# for char dans word, counter < size
#     char = word[counter]
#
#     is char part of the vocab?
#         counter ++
#     if(second char fait partie du vocab)char += word[counter]
#
#     while(counter < nGramSize)
#         check if there is a next chars (depend if bi-gram or tri-gram)
#             char = word[counter];
