import sys

import nltk, re, pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import names
import enchant
import codecs
from random import shuffle
import becas
becas.email = 'tbaldy123@gmail.com'
becas.tool = 'gene-disease-relationship-finder'

def customNamedEntities(txtFileName):
	## Test named entity extractor on a file
	if txtFileName != None:
		# Open text file
		f = open(txtFileName, "r")

		# Count values for genes and non-genes
		count = 0

		# For each line, classify and print result
		for line in f:
			# Handle newline with no text
			if len(line) <= 0:
				continue

			# Classification of word
			results = becas.annotate_text(line, groups={
				"DISO": True,
				"PRGE": True
			})
			print(results)

			count += 1

			if count >= 10:
				break


def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python becasgenesdiseases.py [[<txtfilename>.txt]]")
		return 1

	# Text file to run the gene classifier on
	textFileName = None
	if len(sys.argv) >= 2:
		textFileName = sys.argv[1]
		if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
			print("Invalid file name.")
			print("Format: python becasgenesdiseases.py [[<txtfilename>.txt]]")
			return 1

	# customNamedEntities(diseaseFileName, 'gene', notDiseaseFileName, 'not gene', textFileName)

	customNamedEntities(textFileName)
	return 0

if __name__ == "__main__":
	main()
