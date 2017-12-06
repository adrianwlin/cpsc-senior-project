import re
import nltk
import sys
from nltk.corpus import ieer
from nltk.sem import relextract


def findRelations(filename):
    with open(filename, "r") as f:
        contents = f.readlines()
    for abstract in contents:
        sentences = nltk.sent_tokenize(abstract)
        # print len(sentences)
        tokenized_sentences = [nltk.word_tokenize(
            sentence) for sentence in sentences]
        tagged_sentences = [nltk.pos_tag(sentence)
                            for sentence in tokenized_sentences]


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
