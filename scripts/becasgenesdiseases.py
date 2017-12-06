import sys

import nltk
import re
import pprint
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import names
import enchant
import codecs
import pickle
import os
from random import shuffle
import becas
from os.path import dirname
from customnamedentitiesgenes import customNamedEntities
becas.email = 'tbaldy123@gmail.com'
becas.tool = '490-gene-disease-relationship-finder'

BATCH_SIZE = 1000


def removeNonAscii(s):
    """
    https://stackoverflow.com/questions/1342000/how-to-make-the-python-interpreter-correctly-handle-non-ascii-characters-in-stri
    """
    return "".join(i for i in s if ord(i) < 128)


def createDiseaseEntry(diso, sent):
    token = diso.split('|')[0]
    dis = {}
    dis['index'] = sent.find(token)
    dis['lengthInChars'] = len(token)
    dis['lengthInWords'] = len(token.split(' '))
    dis['name'] = token

    # Add the CUI code to the output data
    codes = diso.split('|')[1].split(':')
    if len(codes[1]) == 8 and codes[1][0] == 'C':
        dis['cui'] = codes[1]
    else:
        dis['cui'] = ''
        for elem in codes:
            if len(elem) == 8 and elem[0] == 'C':
                dis['cui'] = elem
                break
    return dis


def becasNER(txtFileName, start_line=0, geneFileName=None, notGeneFileName=None):
    """
    Runs becas NER for genes and diseases on each sentence. Writes to a new pickle
    file for every 1000 sentences.
    """
    d = enchant.Dict("en_US")  # English Dictionary
    output = []
    fullOutput = []
    # Test named entity extractor on a file
    if txtFileName != None:
        # Open text file
        f = open(txtFileName, "r")

        # Output format
        # [{
        # 	line: string, # Full sentence
        # 	genes: [{
        # 				index: int, # Index into sentence
        # 				lengthInChars: int, # Length of gene name
        # 				lengthInWords: int, # Length of gene name
        # 				name: string, # Full gene name
        # 				uniprot: string # UNIPROT code of gene
        # 			}],
        # 	diseases: [{
        # 				index: int, # Index into sentence
        # 				lengthInChars: int, # Length of disease name
        # 				lengthInWords: int, # Length of disease name
        # 				name: string, # Full disease name
        # 				cui: string # Disease Concept Unique Identifier
        # 			}]
        # }, {
        # 	...
        # }
        # ...]

        files_written = start_line / BATCH_SIZE
        sentences_processed = 0
        # For each line, classify and print result
        lines = f.readlines()
        # numbeer of lines from the file that have been read
        lines_read = 0
        lines_to_read = len(lines) - start_line
        for line in lines[start_line:]:
            line = removeNonAscii(line)
            for sent in nltk.tokenize.sent_tokenize(line):
                # Initialize data (the next entry in output)
                data = {}
                data['line'] = sent
                data['genes'] = []
                data['diseases'] = []

                # Handle newline with no text
                if len(sent) <= 1:
                    continue

                sent = unicode(sent, errors='ignore')

                # Proteins and Genes in sentence
                results_prge = becas.annotate_text(sent, groups={
                    "PRGE": True
                })['entities']

                # Diseases in sentence
                results_diso = becas.annotate_text(sent, groups={
                    "DISO": True
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
                        classifier = customNamedEntities(
                            geneFileName, 'gene', notGeneFileName, 'protein')
                        cl = classifier.classify({'len': len(token),
                                                  'cap_frac': (sum(c.isupper() for c in token) + 0.0) / len(token),
                                                  'num_frac': (sum(c.isdigit() for c in token) + 0.0) / len(token),
                                                  'dict': d.check(token)})
                        if cl == 'gene':
                            data['genes'].append(gene)
                    else:
                        data['genes'].append(gene)

                    # Add the CUI code to the output data
                    codes = prge.split('|')[1].split(':')
                    if len(codes[1]) == 6 and codes[1][0] == 'P':
                        gene['uniprot'] = codes[1]
                    else:
                        gene['uniprot'] = ''
                        for elem in codes:
                            if len(elem) == 6 and elem[0] == 'P':
                                gene['uniprot'] = elem
                                break

                # Add the gene name to data
                for diso in results_diso:
                    dis = createDiseaseEntry(diso, sent)
                    # Update data
                    data['diseases'].append(dis)
                # Update output
                output.append(data)
                fullOutput.append(data)

                sentences_processed += 1
                if sentences_processed % BATCH_SIZE == 0 and sentences_processed > 0:
                    dirname = os.path.dirname(txtFileName)
                    if not os.path.exists(os.path.join(dirname, "becas_labeled")):
                        os.makedirs(os.path.join(dirname, "becas_labeled"))
                    name, _ = os.path.splitext(txtFileName)
                    basename = os.path.basename(name)
                    pickle_filename = "{}_becas_{}.p".format(
                        basename, str(files_written))
                    whole_filename = os.path.join(
                        dirname, "becas_labeled", pickle_filename)
                    with open(whole_filename, "wb") as f:
                        pickle.dump(output, f)
                    print "{} written".format(whole_filename)
                    output = []
                    files_written += 1
            lines_read += 1
            if lines_read % 10 == 0:
                print "{}/{} lines read".format(lines_read, lines_to_read)

    return fullOutput

def main():
    # Check correct number of arguments
    if len(sys.argv) < 2:
        """
        txt file to read, line to start at
        """
        print(
            "Format: python becasgenesdiseases.py <txtfilename>.txt [start_line]")
        return 1

    # Text file to run the gene classifier on
    textFileName = sys.argv[1]
    if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
        print("Invalid text file name.")
        print(
            "Format: python becasgenesdiseases.py <txtfilename>.txt [start_line]")
        return 1

    # If no gene file and non-gene file to train on, run becasNER
    # without differing between proteins and genes
    # Otherwise, also take this into account to differentiate proteins and genes
    if len(sys.argv) == 2:
        entityList = becasNER(textFileName)
    elif len(sys.argv) == 3:
        start_line = int(sys.argv[2])
        entityList = becasNER(textFileName, start_line=start_line)
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
        entityList = becasNER(
            textFileName, geneFileName=geneFileName, nonGeneFileName=nonGeneFileName)

    # Dump this object into a pickle file for Relationship Extractor to use
	pickleDumpFile = textFileName
	# Open pickleDumpFile for writing and dump
	f = open(pickleDumpFile,'wb')
	pickle.dump(entityList,f)
	f.close()

	## Example code for reading pickle file
	# f = open(pickleDumpFile,'r')  
	# testLoad = pickle.load(f)
	# print(testLoad == entityList)

    genesFoundFile = open(textFileName[:-4] + 'genesFound.txt', 'wb')
    diseasesFoundFile = open(textFileName[:-4] + 'diseasesFound.txt', 'wb')

    for line in entityList:
        for gene in line['genes']:
            genesFoundFile.write(gene['name'] + '\n')
        for disease in line['diseases']:
            diseasesFoundFile.write(disease['name'] + '\n')

    genesFoundFile.close()
    diseasesFoundFile.close()

    '''
	The following is all test code to determine accuracy.
	It should only be run on files consisting *only* of gene and disease names.
	'''
    geneCount = 0
    diseaseCount = 0
    wrongGenesCount = 0
    wrongDiseasesCount = 0

    for line in entityList:
        geneCount += len(line['genes'])

        for gene in line['genes']:
            if len(gene['name'].split(' is an entity. ')) > 1:
                print 'found wrong gene: ' + gene['name']
                wrongGenesCount += 1

        diseaseCount += len(line['diseases'])

        for disease in line['diseases']:
            if len(disease['name'].split(' is an entity. ')) > 1:
                print 'found wrong disease: ' + disease['name']
                wrongDiseasesCount += 1

    print 'Total genes found is:'
    print geneCount
    print 'Total disease found is:'
    print diseaseCount
    print 'Also found this many wrong genes:'
    print wrongGenesCount
    print 'Also found this many wrong diseases:'
    print wrongDiseasesCount

    return 0


if __name__ == "__main__":
    main()
