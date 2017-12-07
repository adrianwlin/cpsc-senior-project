import sys

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree, ParentedTree
from nltk.corpus import names
import spacy
nlp = spacy.load('en')

DEP_DEPTH_CAP = 50

# Stole this from https://stackoverflow.com/questions/1342000/how-to-make-the-python-interpreter-correctly-handle-non-ascii-characters-in-stri
def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def depParse(text, index1, index2):
	# Check if indices are valid
	if index1 < 0 or index1 >= len(text.split(' ')) or index2 <= 0 or index2 >= len(text.split(' ')):
		return None

	depFeats = [] # Dependency features
	try:
		text = unicode(text, errors='ignore') # Make sure text is unicode
	except:
		text = text
	doc = nlp(text) # run spaCy nlp on text

	# def to_nltk_tree(node):
	# 	if node.n_lefts + node.n_rights > 0:
	# 		return ParentedTree(node.orth_, [to_nltk_tree(child) for child in node.children])
	# 	else:
	# 		return ParentedTree(node.orth_, [])

	# # exSents = nlp(unicode('Because I am hungry, I went to eat and play', errors='ignore')).sents
	# for sent in doc.sents:
	# 	print to_nltk_tree(sent.root).pretty_print()

	# Extract the dependency features
	for token in doc:
		depFeats.append({
			'text': token.text,
			'pos': token.pos_,
			'dep': token.dep_,
			'head': token.head.text,
			'headpos': token.head.pos_,
			'children': [unicode(child) for child in token.children]
			});

	'''
	This function returns a list of all the direct ancestors of a node.
	Input: index of node in the depFeats list.
	'''
	def parents(index):
		dep = None
		text = depFeats[index]['text']
		head = depFeats[index]['head']
		headpos = depFeats[index]['headpos']

		parents = [] # Output all the ancestors

		loopcount = 0
		while dep != 'ROOT':
			# Put a finite loop count because for some reason
			# There can be loops of parents 4+ items long?
			if loopcount == DEP_DEPTH_CAP:
				# Probably an infinite loop
				# Who rights a sentence with a dependency tree of depth more than 50?
				return None

			loopcount += 1

			# print  "DEPS ARE:"
			# print dep
			foundParent = None
			for item in depFeats:
				if item['text'] == head and item['pos'] == headpos and (text in item['children'] or text == head):
					if text == head and dep != 'ROOT':
						return None
					# Found the parent
					parents.append(item)
					foundParent = item
					break

			dep = foundParent['dep']
			text = foundParent['text']
			head = foundParent['head']
			headpos = foundParent['headpos']

		# print "FULL LIST OF PARENTS ARE:"
		# print parents
		return parents

	oneParents = parents(index1)
	twoParents = parents(index2)

	# Error finding parents (likely due to bad parse of sentence)
	if oneParents == None or twoParents == None:
		return None

	output = {}
	output['dependencyTagOne'] = depFeats[index1]['dep']
	output['dependencyTagTwo'] = depFeats[index2]['dep']
	output['distance'] = abs(index1 - index2)

	LCS = None # Initialize lowest common subsumer

	## Calculate the dependency tree distance of one and two
	output['treeDistance'] = -1
	for i in range(len(oneParents)):
		# All of twoParents are in oneParents
		if i >= len(twoParents):
			output['treeDistance'] = len(oneParents) - len(twoParents) + 2

			# LCS is first in twoParents
			LCS = twoParents[0]
			break

		# Current parents not the same
		if oneParents[len(oneParents) - i - 1] != twoParents[len(twoParents) - i - 1]:
			output['treeDistance'] = len(oneParents) + len(twoParents) - (2 * i) + 2

			# LCS is the last same parent
			try:
				LCS = oneParents[len(oneParents) - i]
			except:
				# Error finding parents, possibly because two roots for some reason
				return None
			break

	# All of oneParents are in twoParents
	if output['treeDistance'] == -1:
		output['treeDistance'] = len(twoParents) - len(oneParents) + 2
		# LCS is first in oneParents
		LCS = oneParents[0]

	# Fill in LCS-related output values
	try:
		output['dependencyTagLCS'] = LCS['dep']
		output['posLCS'] = LCS['pos']
		output['textLCS'] = LCS['text']
	except:
		return None
			
	'''
	Output format:
	{
		'distance': int, # distance between indices
		'treeDistance': int, # distance in dependency tree
		'dependencyTagOne': unicode, # dependency tag of first word
		'dependencyTagTwo': unicode, # dependency tag of second word
		'dependencyTagLCS': unicode, # dependency tag of lowest common subsumer of one and two
		'posLCS': unicode, # part of speech of lowest common subsumer of one and two
		'textLCS': unicode, # text of lowest common subsumer of one and two
	}

	If None is returned, there was an issue determining the parents of the nodes
	This sentence should then be skipped
	'''
	# print depFeats
	return output

def main():
	out = depParse('The quick brown fox jumps over the lazy dog.', 1, 7)
	if out == None:
		print 'Unable to get dependency parse, or parse empty!'
	else:
		# print "OUT IS"
		print out

	return 0

if __name__ == "__main__":
    main()
