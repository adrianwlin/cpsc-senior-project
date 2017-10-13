import sys

import nltk, re, pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import names
import enchant
import codecs

#### This file is a work in progress! Currently it is just the non-custom one

def customNamedEntities(file1name, label1, file2name, label2, txtFileName=None):
	# File being read
	f1 = open(file1name, "r")
	f2 = open(file2name, "r")

	labeled_names = []
	featuresets = []
	d = enchant.Dict("en_US") # English Dictionary

	for line in f1:
		labeled_names.append((line, label1))

		## Features:
		# Length
		# Number of capital letters out of length
		# Number of numbers out of length
		# Not a dictionary word
		featuresets.append(({'word': line, 'len': len(line), \
			'cap_frac': (sum(map(str.isupper, line)) + 0.0)/len(line), \
			'num_frac': (sum(map(str.isdigit, line)) + 0.0)/len(line), \
			'dict': d.check(line)}, label1))

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

	train_set, test_set = featuresets[len(featuresets)/2:], featuresets[:len(featuresets)/2]
	classifier = nltk.NaiveBayesClassifier.train(train_set)

	print featuresets

	# Test some basic cases
	testword1 = 'A2M'
	testword2 = 'Hello'
	testword3 = '1234'

	print "Classifier classifies A2M as " + classifier.classify({'len': len(testword1), \
				'cap_frac': (sum(map(str.isupper, testword1)) + 0.0)/len(testword1), \
				'num_frac': (sum(map(str.isdigit, testword1)) + 0.0)/len(testword1), \
				'dict': d.check(testword1)})
	print "Classifier classifies Hello as " + classifier.classify({'len': len(testword2), \
				'cap_frac': (sum(map(str.isupper, testword2)) + 0.0)/len(testword2), \
				'num_frac': (sum(map(str.isdigit, testword2)) + 0.0)/len(testword2), \
				'dict': d.check(testword2)})
	print "Classifier classifies 1234 as " + classifier.classify({'len': len(testword3), \
				'cap_frac': (sum(map(str.isupper, testword3)) + 0.0)/len(testword3), \
				'num_frac': (sum(map(str.isdigit, testword3)) + 0.0)/len(testword3), \
				'dict': d.check(testword3)})

	f1.close()
	f2.close()
	
	## Test named entity extractor on a file
	# if txtFileName != None:


def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python customnamedentities.py <genefilename>.txt <notgenefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Get input file name and check validity
	geneFileName = sys.argv[1]
	if len(geneFileName) < 4 or (geneFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python customnamedentities.py <genefilename>.txt <notgenefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Get input file name and check validity
	notGeneFileName = sys.argv[2]
	if len(notGeneFileName) < 4 or (notGeneFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python customnamedentities.py <genefilename>.txt <notgenefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# textFileName = sys.argv[3]
	# if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
	# 	print("Invalid file name.")
	# 	print("Format: python customnamedentities.py <genefilename>.txt [[<txtfilename>.txt]]")
	# 	return 1

	customNamedEntities(geneFileName, 'gene', notGeneFileName, 'not gene', None)
	return 0

if __name__ == "__main__":
	main()