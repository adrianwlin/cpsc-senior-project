import sys

import nltk, re, pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

def extractNamedEntities(filename):
	# File being read
	f = open(filename, "r")

	# Full list of named entities
	fullList = []

	# Loop through file and extract entities in each line
	for line in f:
		# Get the named-entity tree of this line
		ne_tree = ne_chunk(pos_tag(word_tokenize(line)))

		# Initialize variables
		continuous_chunk = [] # Named entities in this line
		current_chunk = [] # Current chunk (part of named-entity)

		# Loop through element of tree and create chunks (parts of named entities)
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

		# Update the full list of named-entities
		fullList += continuous_chunk

	print fullList
	f.close()
	return fullList

def findGenes(words):
	genes = []
	gene_regex = re.compile("^[0-9A-Z]*[A-Z]+[0-9A-Z]*$")
	for word in words:
		if gene_regex.match(word):
			genes.append(word)
	print genes
	return genes

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

	named_entities = extractNamedEntities(txtFileName)
	print "Entities extracted."
	genes = findGenes(named_entities)
	print "Genes extracted."
	return 0

if __name__ == "__main__":
	main()
