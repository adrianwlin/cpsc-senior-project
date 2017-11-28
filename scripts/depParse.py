import sys

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import names
import spacy
nlp = spacy.load('en')

# Stole this from https://stackoverflow.com/questions/1342000/how-to-make-the-python-interpreter-correctly-handle-non-ascii-characters-in-stri
def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def depParse(text):
	output = []
	text = unicode(text, errors='ignore')
	doc = nlp(text)
	for token in doc:
		output.append({
			'text': token.text,
			'dep': token.dep_,
			'head': token.head.text,
			'headpos': token.head.pos_,
			'children': [child for child in token.children]
			});
			
	return output

def main():
	print depParse('The quick brown fox jumped over the lazy dog.')

	return 0

if __name__ == "__main__":
    main()
