import sys

import nltk, re, pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import names
import enchant
import codecs
from random import shuffle

def customNamedEntities(file1name, label1, file2name, label2, txtFileName=None):
	# File being read
	f1 = open(file1name, "r")
	f2 = open(file2name, "r")

	labeled_names = []
	featuresets = []
	d = enchant.Dict("en_US") # English Dictionary

	### IMPORTANT NOTE
	# Since diseases can be more than 1 word unlike genes,
	# These are trained 1 per LINE instead of 1 per word token in training file
	for line in f1:
		labeled_names.append((line, label1))

		## Features:
		# Length
		# Number of capital letters out of length
		# Number of numbers out of length
		# Not a dictionary word
		featuresets.append(({'line': line, 'len': len(line), \
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

	shuffle(featuresets)
	train_set, test_set = featuresets[len(featuresets)/2:], featuresets[:len(featuresets)/2]
	classifier = nltk.NaiveBayesClassifier.train(train_set)

	# print featuresets

	# Test some basic cases
	# testword1 = 'A2M'
	# testword2 = 'Hello'
	# testword3 = '1234'

	# print "Classifier classifies A2M as " + classifier.classify({'len': len(testword1), \
	# 			'cap_frac': (sum(map(str.isupper, testword1)) + 0.0)/len(testword1), \
	# 			'num_frac': (sum(map(str.isdigit, testword1)) + 0.0)/len(testword1), \
	# 			'dict': d.check(testword1)})
	# print "Classifier classifies Hello as " + classifier.classify({'len': len(testword2), \
	# 			'cap_frac': (sum(map(str.isupper, testword2)) + 0.0)/len(testword2), \
	# 			'num_frac': (sum(map(str.isdigit, testword2)) + 0.0)/len(testword2), \
	# 			'dict': d.check(testword2)})
	# print "Classifier classifies 1234 as " + classifier.classify({'len': len(testword3), \
	# 			'cap_frac': (sum(map(str.isupper, testword3)) + 0.0)/len(testword3), \
	# 			'num_frac': (sum(map(str.isdigit, testword3)) + 0.0)/len(testword3), \
	# 			'dict': d.check(testword3)})

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
					'cap_frac': (sum(map(str.isupper, token)) + 0.0)/len(token), \
					'num_frac': (sum(map(str.isdigit, token)) + 0.0)/len(token), \
					'dict': d.check(testword1)})

				# Print output
				# print "Classifier classifies " + token + " as " + cl

				# Increment count of "gene" and "not gene"
				if cl == "gene":
					g += 1
				else:
					ng +=1

		print "Total number of genes classified is: " + str(g)
		print "Total number of non-genes classified is: " + str(ng)


def main():
	# Check correct number of arguments
	if len(sys.argv) < 3:
		print("Format: python customnamedentitiesdiseases.py <diseasefilename>.txt <notdiseasefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Get input file name and check validity
	diseaseFileName = sys.argv[1]
	if len(diseaseFileName) < 4 or (diseaseFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python customnamedentitiesdiseases.py <diseasefilename>.txt <notdiseasefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Get input file name and check validity
	notDiseaseFileName = sys.argv[2]
	if len(notDiseaseFileName) < 4 or (notDiseaseFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python customnamedentitiesdiseases.py <diseasefilename>.txt <notdiseasefilename>.txt [[<txtfilename>.txt]]")
		return 1

	# Text file to run the gene classifier on
	textFileName = None
	if len(sys.argv) >= 4:
		textFileName = sys.argv[3]
		if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
			print("Invalid file name.")
			print("Format: python customnamedentitiesdiseases.py <diseasefilename>.txt [[<txtfilename>.txt]]")
			return 1

	customNamedEntities(diseaseFileName, 'gene', notDiseaseFileName, 'not gene', textFileName)
	return 0

if __name__ == "__main__":
	main()
