import sys

import nltk, re, pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import names
import enchant
import codecs
from random import shuffle

'''
Our own custom named-entity recognizer that uses syntax to determine if something is a gene.
Takes in a file of things with label1 (we use this for a file of genes),
a file of things with label2 (we use this for a file of non-gene words), and an optional txtFile to analyze.
Returns the classifier that can be used to classify things as label1 or label2 (gene or notgene).
'''
def customNamedEntities(file1name, label1, file2name, label2, txtFileName=None):
	# File being read
	f1 = open(file1name, "r")
	f2 = open(file2name, "r")

	labeled_names = []
	featuresets = []
	d = enchant.Dict("en_US") # English Dictionary

	for line in f1:
		for word in word_tokenize(line.decode('utf-8')):
			labeled_names.append((word, label1))

			## Features:
			# Length
			# Number of capital letters out of length
			# Number of numbers out of length
			# Not a dictionary word
			featuresets.append(({'word': word, 'len': len(word), \
				'cap_frac': (sum(c.isupper() for c in word) + 0.0)/len(word), \
				'num_frac': (sum(c.isdigit() for c in word) + 0.0)/len(word), \
				'dict': d.check(word)}, label1))

	for line in f2:
		for word in word_tokenize(line.decode('utf-8')):
			labeled_names.append((word, label2))

			## Features:
			# Length
			# Number of capital letters out of length
			# Number of numbers out of length
			# Not a dictionary word
			featuresets.append(({'word': word, 'len': len(word), \
				'cap_frac': (sum(c.isupper() for c in word) + 0.0)/len(word), \
				'num_frac': (sum(c.isdigit() for c in word) + 0.0)/len(word), \
				'dict': d.check(word)}, label2))

	shuffle(featuresets)
	train_set, test_set = featuresets[len(featuresets)/2:], featuresets[:len(featuresets)/2]
	classifier = nltk.NaiveBayesClassifier.train(train_set)

	f1.close()
	f2.close()
	
	## Test named entity extractor on a file
	if txtFileName != None:
		# Open text file
		f3 = open(txtFileName, "r")

		# Count values for genes and non-genes
		g = 0
		ng = 0

		# For each line, classify and print result
		for line in f3:
			# Each word in the line
			for token in line.split():
				# Handle newline with no text
				if len(token) <= 0:
					continue

				# Classification of word
				cl = classifier.classify({'len': len(token), \
					'cap_frac': (sum(c.isupper() for c in token) + 0.0)/len(token), \
					'num_frac': (sum(c.isdigit() for c in word) + 0.0)/len(token), \
					'dict': d.check(token)})

				# Print output
				# print "Classifier classifies " + token + " as " + cl

				# Increment count of "gene" and "not gene"
				if cl == "gene":
					g += 1
				else:
					ng +=1

		print "Total number of genes classified is: " + str(g)
		print "Total number of non-genes classified is: " + str(ng)
	return classifier


def main():
	# Check correct number of arguments
	if len(sys.argv) < 3:
		print("Format: python customnamedentitiesgenes.py <genefilename>.txt <notgenefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Get input file name and check validity
	geneFileName = sys.argv[1]
	if len(geneFileName) < 4 or (geneFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python customnamedentitiesgenes.py <genefilename>.txt <notgenefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Get input file name and check validity
	notGeneFileName = sys.argv[2]
	if len(notGeneFileName) < 4 or (notGeneFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python customnamedentitiesgenes.py <genefilename>.txt <notgenefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Text file to run the gene classifier on
	textFileName = None
	if len(sys.argv) >= 4:
		textFileName = sys.argv[3]
		if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
			print("Invalid file name.")
			print("Format: python customnamedentitiesgenes.py <genefilename>.txt [[<txtfilename>.txt]]")
			return 1

	customNamedEntities(geneFileName, 'gene', notGeneFileName, 'not gene', textFileName)
	return 0

if __name__ == "__main__":
	main()