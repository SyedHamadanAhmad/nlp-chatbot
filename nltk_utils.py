import nltk
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


words=["organize", "organizing", "organization"]
stemmed_words=[stem(w) for w in words]

print(stemmed_words)