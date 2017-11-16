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

		## Output format
		# [{
		# 	line: string, # Full sentence
		# 	genes: [{
		# 				index: int, # Index into sentence
		# 				length: int, # Length of gene name
		# 				name: string # Full gene name
		# 			}],
		# 	diseases: [{
		# 				index: int, # Index into sentence
		# 				length: int, # Length of gene name
		# 				name: string # Full gene name
		# 			}]
		# }, {
		# 	...
		# }
		# ...]
		output = []

		# For each line, classify and print result
		for line in f:
			for sent in nltk.tokenize.sent_tokenize(line):
				data = {}
				data['line'] = sent
				data['genes'] = []
				data['diseases'] = []

				# Handle newline with no text
				if len(sent) <= 1:
					continue

				# Proteins and Genes in sentence
				results_prge = becas.annotate_text(sent, groups={
					"PRGE": True
				})['entities']

				# Add the gene name to data
				for prge in results_prge:
					data['genes'].append(prge.split('|')[0])

				# Diseases in sentence
				results_diso = becas.annotate_text(sent, groups={
					"DISO": True
				})['entities']

				# Add the gene name to data
				for diso in results_diso:
					data['diseases'].append(diso.split('|')[0])

				output.append(data)
	return output


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

	print(customNamedEntities(textFileName))
	return 0

if __name__ == "__main__":
	main()
