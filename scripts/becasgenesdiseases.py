import sys

import nltk, re, pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import names
import enchant
import codecs
from random import shuffle
import becas
from customnamedentitiesgenes import customNamedEntities
becas.email = 'tbaldy123@gmail.com'
becas.tool = 'gene-disease-relationship-finder'

def becasNER(txtFileName, geneFileName=None, notGeneFileName=None):
	d = enchant.Dict("en_US") # English Dictionary
	## Test named entity extractor on a file
	if txtFileName != None:
		# Open text file
		f = open(txtFileName, "r")

		## Output format
		# [{
		# 	line: string, # Full sentence
		# 	genes: [{
		# 				index: int, # Index into sentence
		# 				lengthInChars: int, # Length of gene name
		# 				lengthInWords: int, # Length of gene name
		# 				name: string # Full gene name
		# 			}],
		# 	diseases: [{
		# 				index: int, # Index into sentence
		# 				lengthInChars: int, # Length of disease name
		# 				lengthInWords: int, # Length of disease name
		# 				name: string # Full disease name
		# 				cui: string # Disease Concept Unique Identifier
		# 			}]
		# }, {
		# 	...
		# }
		# ...]
		output = []

		# For each line, classify and print result
		for line in f:
			for sent in nltk.tokenize.sent_tokenize(line):
				# Initialize data (the next entry in output)
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
					token = prge.split('|')[0]
					gene = {}
					gene['index'] = sent.find(token)
					gene['lengthInChars'] = len(token)
					gene['lengthInWords'] = len(token.split(' '))
					gene['name'] = token

					# Check if protein or gene if training data given
					if geneFileName != None and notGeneFileName != None:
						classifier = customNamedEntities(geneFileName, 'gene', notGeneFileName, 'protein')
						cl = classifier.classify({'len': len(token), \
								'cap_frac': (sum(map(str.isupper, token)) + 0.0)/len(token), \
								'num_frac': (sum(map(str.isdigit, token)) + 0.0)/len(token), \
								'dict': d.check(token)})
						if cl == 'gene':
							data['genes'].append(gene)
					else:
						data['genes'].append(gene)

				# Diseases in sentence
				results_diso = becas.annotate_text(sent, groups={
					"DISO": True
				})['entities']

				print "results_diso are: "
				print results_diso

				# Add the gene name to data
				for diso in results_diso:
					token = diso.split('|')[0]
					dis = {}
					dis['index'] = sent.find(token)
					dis['lengthInChars'] = len(token)
					dis['lengthInWords'] = len(token.split(' '))
					dis['name'] = token

					# Add the CUI code to the output data
					codes = diso.split('|')[1].split(':')[1]
					if len(codes[1] == 8 and codes[1][0] == 'C'):
						dis['cui'] = codes[1]
					else:
						dis['cui'] = ''
						for elem in codes:
							if len(elem == 8 and elem[0] == 'C'):
								dis['cui'] = elem
								break

					# Update data
					data['diseases'].append(dis)

				# Update output
				output.append(data)
	return output


def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python becasgenesdiseases.py <txtfilename>.txt")
		return 1

	# Text file to run the gene classifier on
	textFileName = None
	if len(sys.argv) >= 2:
		textFileName = sys.argv[1]
		if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
			print("Invalid text file name.")
			print("Format: python becasgenesdiseases.py <txtfilename>.txt")
			return 1
	
	# If no gene file and non-gene file to train on, run becasNER
	# without differing between proteins and genes
	# Otherwise, also take this into account to differentiate proteins and genes
	if len(sys.argv) < 4:
		print(becasNER(textFileName))
	else:
		geneFileName = sys.argv[2]
		if len(geneFileName) < 4 or (geneFileName[-4:] != ".txt"):
			print("Invalid gene file name.")
			print("Format: python becasgenesdiseases.py <txtfilename>.txt <genefilename>.txt <non-genefilename>.txt")
			return 1

		nonGeneFileName = sys.argv[3]
		if len(nonGeneFileName) < 4 or (nonGeneFileName[-4:] != ".txt"):
			print("Invalid non-gene file name.")
			print("Format: python becasgenesdiseases.py <txtfilename>.txt <genefilename>.txt <non-genefilename>.txt")
			return 1

		print(becasNER(textFileName, geneFileName, nonGeneFileName))
	return 0

if __name__ == "__main__":
	main()
