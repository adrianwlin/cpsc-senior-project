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
import string


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


class Preprocessor:
    def __init__(self, minDistance=-30, maxDistance=30):
        # This dict maps each distance value in range [-30, 30] on to a positive value.
        # The values 0, 1, 2 are reserved for special cases such has padding or out of bounds.
        self.distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        for dis in range(self.minDistance, self.maxDistance + 1):
            # {-30: 3, -29: 4, ...etc }
            self.distanceMapping[dis] = len(self.distanceMapping)
        # Build Word2Vec Model
        sentences = MySentences(os.path.join(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))), 'util/word2vectraining'))  # a memory-friendly iterator
        # Train a Word2Vec Model
        # Code from: https://rare-technologies.com/word2vec-tutorial/
        self.word2vec_model = Word2Vec(sentences)

        # Words that were found in the new text but were not in the training corpus
        self.newFoundWords = {}

        # Length of a word2vec entry
        self.w2vlen = 100

    def entitySpacePadding(self, entry):
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
        return sentence

    def getDistances(self, gene_ind, gene_length, disease_ind, disease_length,
                     maxSentenceLen, num_words):
        geneDistances = np.zeros(maxSentenceLen)
        diseaseDistances = np.zeros(maxSentenceLen)

        for i in range(0, min(maxSentenceLen, num_words)):
            # If a gene is two words long, the distance array will look
            # something like [..., -2, -1, 0, 0, 1, 2,...]
            if i < gene_ind:
                geneDistance = i - gene_ind
            elif gene_ind <= i < gene_ind + gene_length:
                geneDistance = 0
            else:
                geneDistance = i - \
                    (gene_ind + gene_length - 1)
            # Do the same thing for disease.
            if i < disease_ind:
                diseaseDistance = i - disease_ind
            elif disease_ind <= i < disease_ind + disease_length:
                diseaseDistance = 0
            else:
                diseaseDistance = i - \
                    (disease_ind + disease_length - 1)
            # Map the distances according to distanceMapping.
            if geneDistance in self.distanceMapping:
                geneDistances[i] = self.distanceMapping[geneDistance]
            elif geneDistance <= self.minDistance:
                geneDistances[i] = self.distanceMapping['LowerMin']
            else:
                geneDistances[i] = self.distanceMapping['GreaterMax']
            if diseaseDistance in self.distanceMapping:
                diseaseDistances[i] = self.distanceMapping[diseaseDistance]
            elif diseaseDistance <= self.minDistance:
                diseaseDistances[i] = self.distanceMapping['LowerMin']
            else:
                diseaseDistances[i] = self.distanceMapping['GreaterMax']
        return geneDistances, diseaseDistances

    def getTokenIndex(self, raw_sentence, raw_index, tokens):
        '''
        tokens is a tokenized version of raw_sentence.
        This function finds the token index corresponding to the raw_index in to the
        string raw_sentence.
        Returns -1 if it cannot be found.
        '''
        offset = 0
        for i, token in enumerate(tokens):
            offset = raw_sentence.find(token, offset)
            if offset <= raw_index < offset + len(token):
                return i
            offset += len(token)
        return -1

    def createTrainingFeatures(self, labeled_list,  maxSentenceLen=100):
        '''
        Create np.array objects for each set of features we want to use in the CNN
        '''
        # Lists for each feature
        labels = []
        geneDistanceMatrix = []
        diseaseDistanceMatrix = []
        wordEmbedMatrix = []
        skipped = 0
        succeeded = 0
        nodepscount = 0

        # Lists for each feature from depParse (Dependency Tree)
        depFeaturesList = []

        for i, entry in enumerate(labeled_list):
            if i % 500 == 0 and i > 0:
                print "{} sentences processed, {} pairs labeled".format(i, succeeded)
            raw_sentence = entry["line"]
            sentence = self.entitySpacePadding(entry)
            # Tokenize the sentence
            words = nltk.word_tokenize(sentence)

            for gene_dis_pair, label in entry["labels"].items():
                gene = gene_dis_pair[0]
                for g in entry["genes"]:
                    if g["name"] == gene:
                        gene_entry = g
                gene_length = gene_entry["lengthInWords"]
                # Get the token index of the first word of the gene
                gene_ind = self.getTokenIndex(
                    raw_sentence, gene_entry["index"], words)
                if gene_ind < 0:
                    skipped += 1
                    continue

                disease = gene_dis_pair[1]
                for d in entry["diseases"]:
                    if d["name"] == disease:
                        disease_entry = d
                disease_length = disease_entry["lengthInWords"]
                # Get the token index of the first word of the disease
                disease_ind = self.getTokenIndex(
                    raw_sentence, disease_entry["index"], words)
                if disease_ind < 0:
                    skipped += 1
                    continue

                # Depedency Parse Features
                dp = depParse(" ".join(words), gene_ind,
                              disease_ind, gene, disease)
                if dp == None:
                    # Error in dependency extraction
                    # Couldn't find intelligent informationa about this
                    depFeaturesList.append(None)
                    nodepscount += 1
                else:
                    # Found the dependency features successfully
                    depFeaturesList.append([dp['distance'], dp['treeDistance'],
                                            int(dp['dependencyTagOne']), int(
                                                dp['dependencyTagTwo']), int(dp['dependencyTagLCS']),
                                            dp['posLCS']])

                # word embeddings for each word in the sentence
                wordEmbeddingIDs = np.zeros((maxSentenceLen, self.w2vlen))

                j = 0  # Index into wordEmbeddingsIDs
                for i in range(0, min(maxSentenceLen, len(words))):
                    # Current word
                    curr = words[i]

                    if curr in self.word2vec_model.wv.vocab:
                        # Known word
                        j += 1
                        try:
                            wordEmbeddingIDs[j] = self.word2vec_model.wv[curr]
                        except KeyError:
                            # Unknown word
                            if curr not in self.newFoundWords:
                                # Completely new word
                                # Parameters based on https://groups.google.com/forum/#!topic/word2vec-toolkit/J3Skqbe3VwQ
                                self.newFoundWords[curr] = np.random.uniform(
                                    -0.25, 0.25, self.w2vlen).tolist()
                            wordEmbeddingIDs[j] = self.newFoundWords[curr]
                    elif all(char in set(string.punctuation) for char in curr) or len(curr) <= 1:
                        # Is punctuation
                        continue
                    else:
                        # Unknown word
                        j += 1
                        if curr not in self.newFoundWords:
                            # Completely new word
                            # Parameters based on https://groups.google.com/forum/#!topic/word2vec-toolkit/J3Skqbe3VwQ
                            self.newFoundWords[curr] = np.random.uniform(
                                -0.25, 0.25, self.w2vlen).tolist()
                        wordEmbeddingIDs[j] = self.newFoundWords[curr]

                geneDistances, diseaseDistances = self.getDistances(
                    gene_ind, gene_length, disease_ind, disease_length,
                    maxSentenceLen, len(words))
                succeeded += 1
                wordEmbedMatrix.append(wordEmbeddingIDs)
                geneDistanceMatrix.append(geneDistances)
                diseaseDistanceMatrix.append(diseaseDistances)
                # boolean to int
                labels.append(int(label))

        # Print how many sentences we had to skip because feature extraction failed
        print "NUMBER OF SKIPPED"
        print skipped
        print "NUMBER WITHOUT DEP"
        print nodepscount
        print "NUMBER SUCCEEDED"
        print succeeded

        # For each (gene, disease) pair, returns the label, word embedding, distance from the gene, distance from the disease,
        # distance between gene and disease, dependency-tree-distance between gene and disease,
        # the dependency tag of each of the gene, disease, and the lowest common subsumer, and the
        # part of speech and text of the lowest common subsumer.
        return np.array(labels, dtype='int32'), np.array(wordEmbedMatrix), \
            np.array(geneDistanceMatrix, dtype='int32'), np.array(
                diseaseDistanceMatrix, dtype='int32'), np.array(depFeaturesList)

    def createFeatures(self, labeled_list,  maxSentenceLen=100):
        # Lists for each feature
        geneDistanceMatrix = []
        diseaseDistanceMatrix = []
        wordEmbedMatrix = []
        skipped = 0
        totalPairs = 0
        succeeded = 0

        # Lists for each feature from depParse (Dependency Tree)
        depFeaturesList = []

        nodepscount = 0

        print len(labeled_list), "entries"
        for i, entry in enumerate(labeled_list):
            if i % 250 == 0 and i > 0:
                print "{} sentences processed, {} pairs succeeded".format(i, succeeded)
            totalPairs += len(entry["genes"]) * len(entry["diseases"])
            # If there are no genes or no disease recognized in the sentence skip.
            if len(entry["genes"]) == 0 or len(entry["diseases"]) == 0:
                continue
            raw_sentence = entry["line"]
            sentence = self.entitySpacePadding(entry)
            # Tokenize the sentence
            words = nltk.word_tokenize(sentence)

            for gene_entry in entry["genes"]:
                gene_name = gene_entry["name"]
                gene_length = gene_entry["lengthInWords"]
                gene_ind = self.getTokenIndex(
                    raw_sentence, gene_entry["index"], words)
                if gene_ind < 0:
                    skipped += 1
                    continue
                for disease_entry in entry["diseases"]:
                    disease_name = disease_entry["name"]
                    disease_length = disease_entry["lengthInWords"]
                    offset = 0
                    disease_ind = self.getTokenIndex(
                        raw_sentence, disease_entry["index"], words)
                    if disease_ind < 0:
                        skipped += 1
                        continue

                    # Depedency Parse Features
                    dp = depParse(" ".join(words), gene_ind,
                                  disease_ind, gene_name, disease_name)
                if dp == None:
                    # Error in dependency extraction
                    # Couldn't find intelligent informationa about this
                    depFeaturesList.append(None)
                    nodepscount += 1
                else:
                    # Found the dependency features successfully
                    depFeaturesList.append([dp['distance'], dp['treeDistance'],
                                            int(dp['dependencyTagOne']), int(
                                                dp['dependencyTagTwo']), int(dp['dependencyTagLCS']),
                                            dp['posLCS']])

                    # word embeddings for each word in the sentence
                    wordEmbeddingIDs = np.zeros((maxSentenceLen, self.w2vlen))

                    j = 0  # Index into wordEmbeddingsIDs
                    for i in range(0, min(maxSentenceLen, len(words))):
                        # Current word
                        curr = words[i]

                        if curr in self.word2vec_model.wv.vocab:
                            # Known word
                            j += 1
                            try:
                                wordEmbeddingIDs[j] = self.word2vec_model.wv[curr]
                            except KeyError:
                                # Unknown word
                                if curr not in self.newFoundWords:
                                    # Completely new word
                                    # Parameters based on https://groups.google.com/forum/#!topic/word2vec-toolkit/J3Skqbe3VwQ
                                    self.newFoundWords[curr] = np.random.uniform(
                                        -0.25, 0.25, self.w2vlen).tolist()
                                wordEmbeddingIDs[j] = self.newFoundWords[curr]
                        elif all(char in set(string.punctuation) for char in curr) or len(curr) <= 1:
                            # Is punctuation
                            continue
                        else:
                            # Unknown word
                            j += 1
                            if curr not in self.newFoundWords:
                                # Completely new word
                                # Parameters based on https://groups.google.com/forum/#!topic/word2vec-toolkit/J3Skqbe3VwQ
                                self.newFoundWords[curr] = np.random.uniform(
                                    -0.25, 0.25, self.w2vlen).tolist()
                            wordEmbeddingIDs[j] = self.newFoundWords[curr]

                    geneDistances, diseaseDistances = self.getDistances(
                        gene_ind, gene_length, disease_ind, disease_length,
                        maxSentenceLen, len(words))
                    succeeded += 1
                    wordEmbedMatrix.append(wordEmbeddingIDs)
                    geneDistanceMatrix.append(geneDistances)
                    diseaseDistanceMatrix.append(diseaseDistances)
        # Print how many sentences we had to skip because feature extraction failed
        print "NUMBER OF SKIPPED"
        print skipped
        print "NUMBER WITHOUT DEP"
        print nodepscount
        print "NUMBER SUCCEEDED"
        print succeeded

        # For each (gene, disease) pair, returns the label, word embedding, distance from the gene, distance from the disease,
        # distance between gene and disease, dependency-tree-distance between gene and disease,
        # the dependency tag of each of the gene, disease, and the lowest common subsumer, and the
        # part of speech and text of the lowest common subsumer.
        return np.array(wordEmbedMatrix, dtype='float64'), \
            np.array(geneDistanceMatrix, dtype='float64'), np.array(
                diseaseDistanceMatrix, dtype='float64'), np.array(depFeaturesList, dtype='float64')


def main():
    print "Load dataset"
    root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(root_folder, "data/round2/labeled.p"), "r") as f:
        labeled = pickle.load(f)
    split_ind = int(len(labeled) * 0.8)
    print "Train size: {}, Test size: {}".format(split_ind, len(labeled) - split_ind)
    train_labeled = labeled[:split_ind]
    test_labeled = labeled[split_ind:]

    maxSentenceLen = 0

    for entry in labeled:
        tokens = entry["line"].split(" ")
        maxSentenceLen = max(maxSentenceLen, len(tokens))
    print "Max Sentence Length: ", maxSentenceLen

    # :: Create token matrices ::
    preprocessor = Preprocessor()
    train_set = preprocessor.createTrainingFeatures(
        train_labeled, maxSentenceLen)
    test_set = preprocessor.createTrainingFeatures(test_labeled, maxSentenceLen)
    with open(os.path.join(root_folder, "data/round2/preprocessed_fixed.p"), "wb") as f:
        pickle.dump(train_set, f, -1)
        pickle.dump(test_set, f, -1)


if __name__ == "__main__":
    main()
