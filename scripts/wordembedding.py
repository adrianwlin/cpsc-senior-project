import sys

import nltk
from gensim.models import Word2Vec
import spacy
nlp = spacy.load('en')

# Stole this from https://stackoverflow.com/questions/1342000/how-to-make-the-python-interpreter-correctly-handle-non-ascii-characters-in-stri
def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def embed(text):

def main():
	print(embed("hello"))

	return 0

if __name__ == "__main__":
    main()
