"""
Based on https://github.com/UKPLab/deeplearning4nlp-tutorial
Based on paper:
Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014,
Relation Classification via Convolutional Deep Neural Network
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

import keras
import os
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Activation, Flatten, concatenate, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model

from keras.utils import np_utils

class RelationCNN:
    def __init__(self, useEmbed = False, useDep = False):
        self.batch_size = 64
        self.n_filters = 100
        self.filter_length = 3
        self.hidden_dims = 100
        self.nb_epoch = 100
        self.position_dims = 50

        self.yTrain, self.wordEmbedTrain, self.geneDistTrain, self.diseaseDistTrain, \
            self.wordDistTrain, self.treeDistTrain, \
            self.geneTagTrain, self.diseaseTagTrain, self.lcsTagTrain, self.lcsPosTrain, self.lcsTextTrain \
            = None, None, None, None, None, None, None, None, None, None, None
        self.yTest, self.wordEmbedTest, self.geneDistTest, self.diseaseDistTest, \
            self.wordDistTest, self.treeDistTest, \
            self.geneTagTest, self.diseaseTagTest, self.lcsTagTest, self.lcsPosTest, self.lcsTextTest \
            = None, None, None, None, None, None, None, None, None, None, None
        self.max_position = None
        self.n_out = None
        self.train_y_cat = None
        self.model = None
        self.max_prec, self.max_rec, self.max_acc, self.max_f1 = 0, 0, 0, 0
        self.useEmbed = useEmbed
        self.useDep = useDep

    def loadDataset(self):
        print "Load dataset"
        root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(root_folder, "data/round2/preprocessed.p"), "r") as f:
            self.yTrain, self.wordEmbedTrain, self.geneDistTrain, self.diseaseDistTrain, self.wordDistTrain, self.treeDistTrain, \
                self.geneTagTrain, self.diseaseTagTrain, self.lcsTagTrain, self.lcsPosTrain, self.lcsTextTrain \
                = pickle.load(f)
            self.yTest, self.wordEmbedTest, self.geneDistTest, self.diseaseDistTest, self.wordDistTrain, self.treeDistTest, \
                self.geneTagTest, self.diseaseTagTest, self.lcsTagTest, self.lcsPosTest, self.lcsTextTest \
                = pickle.load(f)
        #
        self.max_position = max(np.max(self.geneDistTrain), np.max(self.diseaseDistTrain)) + 1
        #
        self.n_out = max(self.yTrain) + 1
        self.train_y_cat = np_utils.to_categorical(self.yTrain, self.n_out)

        print "self.wordEmbedTrain: ", self.wordEmbedTrain.shape
        print "self.geneDistTrain: ", self.geneDistTrain.shape
        print "self.diseaseDistTrain: ", self.diseaseDistTrain.shape
        print "self.yTrain: ", self.yTrain.shape

        print "self.wordEmbedTest: ", self.wordEmbedTest.shape
        print "self.geneDistTest: ", self.geneDistTest.shape
        print "self.diseaseDistTest: ", self.diseaseDistTest.shape
        print "self.yTest: ", self.yTest.shape

        # f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
        # embeddings = pkl.load(f)
        # f.close()
        #
        # print "Embeddings: ", embeddings.shape
        
    def createModel(self):
        geneInput = Input(shape=(self.geneDistTrain.shape[1],))
        geneEmbed = Embedding(input_dim=self.max_position, output_dim=self.position_dims,
                              input_length=self.geneDistTrain.shape[1])(geneInput)

        diseaseInput = Input(shape=(self.diseaseDistTrain.shape[1],))
        diseaseEmbed = Embedding(input_dim=self.max_position, output_dim=self.position_dims,
                                 input_length=self.diseaseDistTrain.shape[1])(diseaseInput)

        wordEmbed = None
        depEmbed = None

        # Word embeddings
        # if self.useEmbed or self.wordEmbedTrain == None:
        #     self.useEmbed = False
        #     wordEmbedInput = Input(shape=(self.wordEmbedTrain.shape[1],self.wordEmbedTrain.shape[2]))
        #     wordEmbedEmbed = Embedding(embeddings.shape[0], embeddings.shape[1],
        #                                  input_length=self.wordEmbedTrain.shape[1], weights=[embeddings], trainable=False)
        # # Dependency stuff
        # if self.useDep:
        #     self.useDep = False
        #     wordDepInput = Input(shape=(self.geneDistTrain.shape[1],))
        #     wordDepEmbed = Embedding(embeddings.shape[0], embeddings.shape[1],
        #                                  input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False)
                             
        # Train depending on whether to use the word embedding and dependency embedding
        # if wordEmbed == None and depEmbed == None:   
        #     mergedEmbed = concatenate([geneEmbed, diseaseEmbed])
        # elif wordEmbed == None:
        #     mergedEmbed = concatenate([geneEmbed, diseaseEmbed, depEmbed])
        # elif depEmbed = None:
        #     mergedEmbed = concatenate([geneEmbed, diseaseEmbed, wordEmbed])
        # else:
        #     mergedEmbed = concatenate([geneEmbed, diseaseEmbed, wordEmbed, depEmbed])            

        mergedEmbed = concatenate([geneEmbed, diseaseEmbed])
        
        convolution = Conv1D(filters=self.n_filters,
                             kernel_size=self.filter_length,
                             padding='same',
                             activation='tanh')(mergedEmbed)
        # we use standard max over time pooling
        max_pool = GlobalMaxPooling1D()(convolution)
        #
        dropout = Dropout(0.25)(max_pool)
        dense_out = Dense(self.n_out, activation='softmax')(dropout)

        self.model = Model(inputs=[geneInput, diseaseInput], outputs=[dense_out])
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='Adam', metrics=['accuracy'])
        self.model.summary()
        print "Start training"

        return self.model

    def getPrecision(self):
        # Precision for non-vague
        targetLabelCount = 0
        correctTargetLabelCount = 0

        for idx in xrange(len(pred_test)):
            if pred_test[idx] == targetLabel:
                targetLabelCount += 1

                if pred_test[idx] == self.yTest[idx]:
                    correctTargetLabelCount += 1

        if correctTargetLabelCount == 0:
            return 0

        return float(correctTargetLabelCount) / targetLabelCount

    def trainModel(self):
        for epoch in xrange(self.nb_epoch):
            # model.fit([self.wordEmbedTrain, self.geneDistTrain, self.diseaseDistTrain],
            self.model.fit([self.geneDistTrain, self.diseaseDistTrain],
                      self.train_y_cat, batch_size=self.batch_size, verbose=True, epochs=1)
            probs = self.model.predict(
                [self.geneDistTest, self.diseaseDistTest], verbose=True)
            pred_test = np.argmax(probs, axis=1)

            dctLabels = np.sum(pred_test)
            print dctLabels
            totalDCTLabels = np.sum(self.yTest)
            print totalDCTLabels

            acc = np.sum(pred_test == self.yTest) / float(len(self.yTest))
            self.max_acc = max(self.max_acc, acc)
            print "Accuracy: %.4f (max: %.4f)" % (acc, self.max_acc)

            # f1Sum = 0
            # f1Count = 0
            # for targetLabel in xrange(1, max(self.yTest)):
            #     prec = getPrecision(pred_test, self.yTest, targetLabel)
            #     rec = getPrecision(self.yTest, pred_test, targetLabel)
            #     f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            #     f1Sum += f1
            #     f1Count += 1

            # macroF1 = f1Sum / float(f1Count)
            # self.max_f1 = max(self.max_f1, macroF1)
            # print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, self.max_f1)

def main():
    # Shows how this code is run
    rcnn = RelationCNN()
    rcnn.loadDataset()
    rcnn.createModel()
    rcnn.trainModel()

if __name__ == "__main__":
    main()
