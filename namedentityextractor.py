import sys

import nltk, re, pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

def extractNamedEntities(filename):
	f = open(filename, "r")
	for line in f:
		ne_tree = ne_chunk(pos_tag(word_tokenize(line)))
	prev = None
	continuous_chunk = []
	current_chunk = []
	for i in ne_tree:
		if type(i) == Tree:
			current_chunk.append(" ".join([token for token, pos in i.leaves()]))
		elif current_chunk:
			named_entity = " ".join(current_chunk)
			if named_entity not in continuous_chunk:
				continuous_chunk.append(named_entity)
				current_chunk = []
		else:
			continue
	print continuous_chunk
	f.close()

	print "Entities extracted."
	return continuous_chunk

def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python namedentityextractor.py <txtfilename>.txt")
		return 1

	# Get input file name and check validity
	txtFileName = sys.argv[1]
	if len(txtFileName) < 4 or (txtFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python namedentityextractor.py <txtfilename>.txt")
		return 1

	extractNamedEntities(txtFileName)
	return 0

if __name__ == "__main__":
	main()