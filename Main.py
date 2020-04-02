from GaussianNaiveBayesClassifier import Classifier

# Use hashmap to store frequency for each language
# Use hashmap to store totalite de n-grams dedans for each language
# need a hashmap for each language to keep the count of occurence of each n-gram. (6 hashmaps)
# need to remember total number of tweets
# need number of n-grams per language



def main():
    nb = Classifier(2, 2, 0)
    nb.train("practice")
    nb.test("practice")
    print("debug")





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
