"""
Based on https://github.com/UKPLab/deeplearning4nlp-tutorial
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

import os
import pickle
import nltk
from nltk.probability import FreqDist
from gensim.models import Word2Vec
from depParse import depParse


print "Load dataset"
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(os.path.join(root_folder, "data/round2/labeled.p"), "r") as f:
    labeled = pickle.load(f)
split_ind = int(len(labeled) * 0.8)
print "Train size: {}, Test size: {}".format(split_ind, len(labeled) - split_ind)
train_labled = labeled[:split_ind]
test_labeled = labeled[split_ind:]

maxSentenceLen = 0

# This dict maps each distance value in range [-30, 30] on to a positive value.
# The values 0, 1, 2 are reserved for special cases such has padding or out of bounds.
distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in range(minDistance, maxDistance + 1):
    # {-30: 3, -29: 4, ...etc }
    distanceMapping[dis] = len(distanceMapping)


def createMatrices(labeled_list, maxSentenceLen=100):
    '''
    Create np.array objects for each set of features we want to use in the CNN
    '''
    # Lists for each feature
    labels = []
    geneDistanceMatrix = []
    diseaseDistanceMatrix = []
    wordEmbedMatrix = []
    skipped = 0

    # Lists for each feature from depParse (Dependency Tree)
    wordDistanceList = []
    treeDistanceList = []
    tagOneList = []
    tagTwoList = []
    tagLCSList = []
    posLCSList = []
    textLCSList = []

    '''
    Train a Word2Vec Model
    Code from: https://rare-technologies.com/word2vec-tutorial/
    '''
    class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()

    # Build Word2Vec Model
    sentences = MySentences(os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), 'util/word2vectraining'))  # a memory-friendly iterator
    model = Word2Vec(sentences)

    for entry in labeled_list:
        sentence = entry["line"]
        entity_names = []
        for gene in entry["genes"]:
            entity_names.append(gene["name"])
        for dis in entry["diseases"]:
            entity_names.append(dis["name"])
        # Put spaces before and after the entity for cases like "A B-word" where "A B" is one entity.
        for entity in entity_names:
            start = sentence.find(entity)
            end = start + len(entity)
            start_buffer = ""
            end_buffer = ""
            if start > 0 and sentence[start - 1] != " ":
                start_buffer = " "
            if end < len(sentence) and sentence[end] != " ":
                end_buffer = " "
            sentence = sentence[:start] + start_buffer + \
                sentence[start:end] + end_buffer + sentence[end:]
        # Tokenize the sentence
        words = nltk.word_tokenize(sentence)

        for gene_dis_pair, label in entry["labels"].items():
            gene = gene_dis_pair[0]
            # Length of the gene entity
            gene_length = len(gene.split(" "))
            disease = gene_dis_pair[1]
            # Length of the disease entity
            disease_length = len(disease.split(" "))
            try:
                first_word = gene.split(" ")[0]
                gene_ind = words.index(first_word)
            except:
                skipped += 1
                continue
            try:
                first_word = disease.split(" ")[0]
                disease_ind = words.index(first_word)
            except:
                skipped += 1
                continue

            # Depedency Parse Features
            dp = depParse(" ".join(words), gene_ind, disease_ind, gene, disease)
            if dp == None:
                wordDistanceList.append(None)
                treeDistanceList.append(None)
                tagOneList.append(None)
                tagTwoList.append(None)
                tagLCSList.append(None)
                posLCSList.append(None)
                textLCSList.append(None)
            else:
                wordDistanceList.append(dp['distance'])
                treeDistanceList.append(dp['treeDistance'])
                tagOneList.append(dp['dependencyTagOne'])
                tagTwoList.append(dp['dependencyTagTwo'])
                tagLCSList.append(dp['dependencyTagLCS'])
                posLCSList.append(dp['posLCS'])
                textLCSList.append(dp['textLCS'])

            wordEmbeddingIDs = np.zeros(maxSentenceLen)
            geneDistances = np.zeros(maxSentenceLen)
            diseaseDistances = np.zeros(maxSentenceLen)

            for i in range(0, min(maxSentenceLen, len(words))):
                # wordEmbeddingIDs[i] = model.wv[words[i]]
                # If a gene is two words long, the distance array will look
                # something like [..., -2, -1, 0, 0, 1, 2,...]
                if i < gene_ind:
                    geneDistance = i - gene_ind
                elif gene_ind <= i < gene_ind + gene_length:
                    geneDistance = 0
                else:
                    geneDistance = i - (gene_ind + gene_length - 1)
                # Do the same thing for disease.
                if i < disease_ind:
                    diseaseDistance = i - disease_ind
                elif disease_ind <= i < disease_ind + disease_length:
                    diseaseDistance = 0
                else:
                    diseaseDistance = i - (disease_ind + disease_length - 1)
                # Map the distances according to distanceMapping.
                if geneDistance in distanceMapping:
                    geneDistances[i] = distanceMapping[geneDistance]
                elif geneDistance <= minDistance:
                    geneDistances[i] = distanceMapping['LowerMin']
                else:
                    geneDistances[i] = distanceMapping['GreaterMax']
                if diseaseDistance in distanceMapping:
                    diseaseDistances[i] = distanceMapping[diseaseDistance]
                elif diseaseDistance <= minDistance:
                    diseaseDistances[i] = distanceMapping['LowerMin']
                else:
                    diseaseDistances[i] = distanceMapping['GreaterMax']
            wordEmbedMatrix.append(wordEmbeddingIDs)
            geneDistanceMatrix.append(geneDistances)
            diseaseDistanceMatrix.append(diseaseDistances)
            # boolean to int
            labels.append(int(label))

    # Print how many sentences we had to skip because feature extraction failed
    print "NUMBER OF SKIPPED"
    print skipped
    print "NUMBER SUCCEEDED"
    print len(labels)

    # For each (gene, disease) pair, returns the label, word embedding, distance from the gene, distance from the disease,
    # distance between gene and disease, dependency-tree-distance between gene and disease,
    # the dependency tag of each of the gene, disease, and the lowest common subsumer, and the
    # part of speech and text of the lowest common subsumer.
    return np.array(labels, dtype='int32'), np.array(wordEmbedMatrix, dtype='int32'), \
        np.array(geneDistanceMatrix, dtype='int32'), np.array(
            diseaseDistanceMatrix, dtype='int32'), np.array(wordDistanceList), \
        np.array(treeDistanceList), np.array(tagOneList), \
        np.array(tagTwoList), np.array(tagLCSList), \
        np.array(posLCSList), np.array(textLCSList),


for entry in labeled:
    tokens = entry["line"].split(" ")
    maxSentenceLen = max(maxSentenceLen, len(tokens))
print "Max Sentence Length: ", maxSentenceLen

# :: Create token matrices ::
train_set = createMatrices(train_labled, maxSentenceLen)
test_set = createMatrices(test_labeled, maxSentenceLen)
with open(outputFilePath, 'wb') as:
    pickle.dump(train_set, f, -1)
    pickle.dump(test_set, f, -1)
