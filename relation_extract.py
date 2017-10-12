import re
import nltk
import sys
from nltk.corpus import ieer
from nltk.sem import relextract


def findRelations(filename):
    with open(filename, "r") as f:
        contents = f.read()
    sentences = nltk.sent_tokenize(contents)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

    IN = re.compile(r'.*\bin\b(?!\b.+ing)')
    for sent in tagged_sentences:
        # print(sent)
        chunked = nltk.ne_chunk_sents(sent)
        relations = relextract.extract_rels('ORGANIZATION', 'LOCATION', sent, corpus='ace', pattern=IN)
        for rel in relations:
            print(relextract.rtuple(rel))

def main():
    # Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python relation_extract.py <txtfilename>.txt")
		return 1

	# Get input file name and check validity
	txtFileName = sys.argv[1]
	if len(txtFileName) < 4 or (txtFileName[-4:] != ".txt"):
		print("Invalid file name.")
		print("Format: python relation_extract.py <txtfilename>.txt")
		return 1

	findRelations(txtFileName)
	return 0

if __name__ == "__main__":
	main()

# docs = ieer.parsed_docs('NYT_19980315')
# for attr, value in docs[0].__dict__.items():
#         print attr, value
# print type(docs[0])
# tree = docs[0].text
# print type(tree)

# IN = re.compile(r'.*\bin\b(?!\b.+ing)')
# for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
#     for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
#         print(nltk.sem.rtuple(rel))
