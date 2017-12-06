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


print "Load dataset"
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(os.path.join(root_folder, "data/round1/labeled.p"), "r") as f:
    labeled = pickle.load(f)

all_words = {}
maxSentenceLen = 0
labelsDistribution = FreqDist()

distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in range(minDistance, maxDistance + 1):
    # {-30: 3, -29: 4, ...etc }
    distanceMapping[dis] = len(distanceMapping)


def createMatrices(labeled_list, word2Idx, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    geneDistanceMatrix = []
    diseaseDistanceMatrix = []
    wordEmbedMatrix = []
    continued = 0

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

    # Test case for the above
    # print("EMBED OF BREAST IS: ")
    # print(model.wv["breast"])

    for entry in labeled_list:
        sentence = entry["line"]
        genes_diseases = []
        for gene in entry["genes"]:
            genes_diseases.append(gene["name"])
        for dis in entry["diseases"]:
            genes_diseases.append(dis["name"])
        # Put spaces before and after the entity in case.
        for entity in genes_diseases:
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
        # Make multi word genes and diseases one token
        words = nltk.word_tokenize(sentence)
        for entity in genes_diseases:
            if len(entity.split(" ")) > 1:
                split = entity.split(" ")
                first_word = split[0]
                try:
                    ind = words.index(first_word)
                    joined = " ".join(words[ind:ind + len(split)])
                    words[ind:ind + len(split)] = [joined]
                except:
                    pass
        print words

        for gene_dis_pair, label in entry["labels"].items():
            gene = gene_dis_pair[0]
            disease = gene_dis_pair[1]
            try:
                gene_ind = words.index(gene)
            except:
                continued += 1
                continue
            try:
                disease_ind = words.index(disease)
            except:
                continued += 1
                continue

            wordEmbeddingIDs = np.zeros(maxSentenceLen)
            geneDistances = np.zeros(maxSentenceLen)
            diseaseDistances = np.zeros(maxSentenceLen)

            for i in range(0, min(maxSentenceLen, len(words))):
                # wordEmbeddingIDs[i] = model.wv[words[i]]

                geneDistance = i - int(gene_ind)
                diseaseDistance = i - int(disease_ind)

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
        # break

    return np.array(labels, dtype='int32'), np.array(wordEmbedMatrix, dtype='int32'), \
        np.array(geneDistanceMatrix, dtype='int32'), np.array(
            diseaseDistanceMatrix, dtype='int32'),


for entry in labeled:
    tokens = entry["line"].split(" ")
    maxSentenceLen = max(maxSentenceLen, len(tokens))
    for token in tokens:
        all_words[token.lower()] = True

print "Max Sentence Length: ", maxSentenceLen

# :: Read in word embeddings ::
word2Idx = {}
embeddings = []
#
# for line in open(embeddingsPath):
#     split = line.strip().split(" ")
#     word = split[0]
#
#     if len(word2Idx) == 0:  # Add padding+unknown
#         word2Idx["PADDING"] = len(word2Idx)
#         vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
#         embeddings.append(vector)
#
#         word2Idx["UNKNOWN"] = len(word2Idx)
#         vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
#         embeddings.append(vector)
#
#     if split[0].lower() in words:
#         vector = np.array([float(num) for num in split[1:]])
#         embeddings.append(vector)
#         word2Idx[split[0]] = len(word2Idx)
#
# embeddings = np.array(embeddings)
#
# print "Embeddings shape: ", embeddings.shape
# print "Len words: ", len(words)
#
# f = open(embeddingsPklPath, 'wb')
# pickle.dump(embeddings, f, -1)
# f.close()

print len(labeled)
# :: Create token matrix ::
train_set = createMatrices(labeled, word2Idx, maxSentenceLen)
# test_set = createMatrices(labeled, word2Idx, maxSentenceLen)
#
# f = open(outputFilePath, 'wb')
# pickle.dump(train_set, f, -1)
# pickle.dump(test_set, f, -1)
# f.close()

# print "Data written to pickle file"
#
# for label, freq in labelsDistribution.most_common(100):
#     print "%s : %f%%" % (label, 100 * freq / float(labelsDistribution.N()))
