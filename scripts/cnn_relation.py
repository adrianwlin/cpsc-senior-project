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
from keras.callbacks import CSVLogger

from keras.utils import np_utils


class RelationCNN:
    def __init__(self):
        self.batch_size = 64
        self.n_filters = 75
        self.filter_length = 3
        self.nb_epoch = 25
        self.position_dims = 50

        self.yTrain, self.wordEmbedTrain, self.geneDistTrain, self.diseaseDistTrain, \
            self.depFeaturesTrain = None, None, None, None, None
        self.yTest, self.wordEmbedTest, self.geneDistTest, self.diseaseDistTest, \
            self.depFeaturesTest = None, None, None, None, None
        self.max_position = None
        self.n_out = None
        # max results for model with dependency features
        self.max_prec_dep, self.max_rec_dep, self.max_acc_dep, self.max_f1_dep = 0, 0, 0, 0
        # max results for model without dependency features
        self.max_prec_no_dep, self.max_rec_no_dep, self.max_acc_no_dep, self.max_f1_no_dep = 0, 0, 0, 0
        self.model_dep = None
        self.model_no_dep = None

    # https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    def normalize_columns(self, arr):
        '''
        normalize_columns for dependency features which maybe large integers
        '''
        rows, cols = arr.shape
        for col in xrange(cols):
            arr[:, col] /= abs(arr[:, col]).max()
        return arr

    def splitDataDep(self):
        '''splits data into data with dependency features and data without'''
        self.yTrain_no_dep, self.wordEmbedTrain_no_dep, self.geneDistTrain_no_dep, \
            self.diseaseDistTrain_no_dep = [], [], [], []
        self.yTrain_dep, self.wordEmbedTrain_dep, self.geneDistTrain_dep, \
            self.diseaseDistTrain_dep, self.depFeaturesTrain_dep = [], [], [], [], []
        for i, d in enumerate(self.depFeaturesTrain):
            if not d:
                self.yTrain_no_dep.append(self.yTrain[i])
                self.wordEmbedTrain_no_dep.append(self.wordEmbedTrain[i])
                self.geneDistTrain_no_dep.append(self.geneDistTrain[i])
                self.diseaseDistTrain_no_dep.append(self.diseaseDistTrain[i])
            else:
                self.yTrain_dep.append(self.yTrain[i])
                self.wordEmbedTrain_dep.append(self.wordEmbedTrain[i])
                self.geneDistTrain_dep.append(self.geneDistTrain[i])
                self.diseaseDistTrain_dep.append(self.diseaseDistTrain[i])
                self.depFeaturesTrain_dep.append(self.depFeaturesTrain[i])
        # convert to numpy arrays
        self.yTrain_no_dep = np.array(self.yTrain_no_dep, dtype='float64')
        self.wordEmbedTrain_no_dep = np.array(
            self.wordEmbedTrain_no_dep, dtype='float64')
        self.geneDistTrain_no_dep = np.array(
            self.geneDistTrain_no_dep, dtype='float64')
        self.diseaseDistTrain_no_dep = np.array(
            self.diseaseDistTrain_no_dep, dtype='float64')
        self.yTrain_dep = np.array(self.yTrain_dep, dtype='float64')
        self.wordEmbedTrain_dep = np.array(
            self.wordEmbedTrain_dep, dtype='float64')
        self.geneDistTrain_dep = np.array(
            self.geneDistTrain_dep, dtype='float64')
        self.diseaseDistTrain_dep = np.array(
            self.diseaseDistTrain_dep, dtype='float64')
        self.depFeaturesTrain_dep = np.array(
            self.depFeaturesTrain_dep, dtype='float64')
        self.depFeaturesTrain_dep = self.normalize_columns(
            self.depFeaturesTrain_dep)

        self.yTest_no_dep, self.wordEmbedTest_no_dep, self.geneDistTest_no_dep, \
            self.diseaseDistTest_no_dep = [], [], [], []
        self.yTest_dep, self.wordEmbedTest_dep, self.geneDistTest_dep, \
            self.diseaseDistTest_dep, self.depFeaturesTest_dep = [], [], [], [], []
        for i, d in enumerate(self.depFeaturesTest):
            if not d:
                self.yTest_no_dep.append(self.yTest[i])
                self.wordEmbedTest_no_dep.append(self.wordEmbedTest[i])
                self.geneDistTest_no_dep.append(self.geneDistTest[i])
                self.diseaseDistTest_no_dep.append(self.diseaseDistTest[i])
            else:
                self.yTest_dep.append(self.yTest[i])
                self.wordEmbedTest_dep.append(self.wordEmbedTest[i])
                self.geneDistTest_dep.append(self.geneDistTest[i])
                self.diseaseDistTest_dep.append(self.diseaseDistTest[i])
                self.depFeaturesTest_dep.append(self.depFeaturesTest[i])
        self.yTest_no_dep = np.array(self.yTest_no_dep, dtype='float64')
        self.wordEmbedTest_no_dep = np.array(
            self.wordEmbedTest_no_dep, dtype='float64')
        self.geneDistTest_no_dep = np.array(
            self.geneDistTest_no_dep, dtype='float64')
        self.diseaseDistTest_no_dep = np.array(
            self.diseaseDistTest_no_dep, dtype='float64')
        self.yTest_dep = np.array(self.yTest_dep, dtype='float64')
        self.wordEmbedTest_dep = np.array(
            self.wordEmbedTest_dep, dtype='float64')
        self.geneDistTest_dep = np.array(self.geneDistTest_dep, dtype='float64')
        self.diseaseDistTest_dep = np.array(
            self.geneDistTest_dep, dtype='float64')
        self.depFeaturesTest_dep = np.array(
            self.depFeaturesTest_dep, dtype='float64')
        self.depFeaturesTest_dep = self.normalize_columns(
            self.depFeaturesTest_dep)

    def loadDataset(self):
        print "Load dataset"
        root_folder = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(root_folder, "data/round2/preprocessed_fixed.p"), "r") as f:
            self.yTrain, self.wordEmbedTrain, self.geneDistTrain, \
                self.diseaseDistTrain, self.depFeaturesTrain = pickle.load(f)
            self.yTest, self.wordEmbedTest, self.geneDistTest, \
                self.diseaseDistTest, self.depFeaturesTest = pickle.load(f)

        self.max_position = max(np.max(self.geneDistTrain),
                                np.max(self.diseaseDistTrain)) + 1
        self.splitDataDep()

        # number of possible output labels (2)
        self.n_out = max(self.yTrain) + 1
        # convert to categorical
        self.train_y_cat_dep = np_utils.to_categorical(
            self.yTrain_dep, self.n_out)
        self.train_y_cat_no_dep = np_utils.to_categorical(
            self.yTrain_no_dep, self.n_out)

        print "self.wordEmbedTrain: ", self.wordEmbedTrain.shape
        print "self.geneDistTrain: ", self.geneDistTrain.shape
        print "self.diseaseDistTrain: ", self.diseaseDistTrain.shape
        print "self.depFeaturesTrain: ", self.depFeaturesTrain.shape
        print "self.yTrain: ", self.yTrain.shape
        print "With deps: {}, Without deps: {}".format(self.yTrain_dep.shape, self.yTrain_no_dep.shape)

        print "self.wordEmbedTest: ", self.wordEmbedTest.shape
        print "self.geneDistTest: ", self.geneDistTest.shape
        print "self.diseaseDistTest: ", self.diseaseDistTest.shape
        print "self.depFeaturesTest: ", self.depFeaturesTest.shape
        print "self.yTest: ", self.yTest.shape
        print "With deps: {}, Without deps: {}".format(self.yTest_dep.shape[0], self.yTest_no_dep.shape[0])

    def createModel(self, useDep=False):
        '''Creates the CNN with dependency features if useDep = True'''
        # Distance embeddings with convolution
        # The gene and disease embeddings are concatenated
        geneInput = Input(shape=(self.geneDistTrain.shape[1],))
        geneDistEmbed = Embedding(input_dim=self.max_position, output_dim=self.position_dims,
                                  input_length=self.geneDistTrain.shape[1])(geneInput)
        diseaseInput = Input(shape=(self.diseaseDistTrain.shape[1],))
        diseaseDistEmbed = Embedding(input_dim=self.max_position, output_dim=self.position_dims,
                                     input_length=self.diseaseDistTrain.shape[1])(diseaseInput)
        mergedDistEmbed = concatenate([geneDistEmbed, diseaseDistEmbed])
        distConvolution = Conv1D(filters=self.n_filters,
                                 kernel_size=self.filter_length,
                                 padding='same',
                                 activation='tanh')(mergedDistEmbed)
        # standard max over time pooling
        max_pool_dist = GlobalMaxPooling1D()(distConvolution)
        # droupout layer
        dropout_dist = Dropout(0.25)(max_pool_dist)

        # Word Embeddings with convolution
        wordEmbedInput = Input(
            shape=(self.wordEmbedTrain.shape[1], self.wordEmbedTrain.shape[2]))
        wordEmbedConvolution = Conv1D(filters=self.n_filters,
                                      kernel_size=self.filter_length,
                                      padding='same',
                                      activation='tanh')(wordEmbedInput)
        max_pool_wordEmbed = GlobalMaxPooling1D()(wordEmbedConvolution)
        dropout_wordEmbed = Dropout(0.25)(max_pool_wordEmbed)

        # If using dependency features, add it to the fully connected final layer.
        if useDep:
            depFeaturesInput = Input(
                shape=(self.depFeaturesTrain_dep.shape[1],))
            merged = concatenate(
                [dropout_dist, dropout_wordEmbed, depFeaturesInput])
        else:
            merged = concatenate(
                [dropout_dist, dropout_wordEmbed])
        # Final layer to with softmax activation which produces probabilites.
        dense_out = Dense(self.n_out, activation='softmax')(merged)

        if useDep:
            model = Model(
                inputs=[geneInput, diseaseInput,
                        wordEmbedInput, depFeaturesInput],
                outputs=[dense_out])
        else:
            model = Model(
                inputs=[geneInput, diseaseInput, wordEmbedInput],
                outputs=[dense_out])

        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam', metrics=['accuracy'])
        model.summary()
        return model

    def getPrecision(self, pred_test, yTest, targetLabel):
        '''
        Gets the precision for the targetLabel of pred_test
        ie the proportion of targetLabel's in pred_test that match those in yTest.
        '''
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
        print "Start training: With dependency"
        # Run the training self.nb_epoch times and take the max accuracy
        for epoch in xrange(self.nb_epoch):
            # Write the output of each epoch to a csv log file
            dep_csv_logger = CSVLogger('dep' + str(epoch) + '.log')
            self.model_dep.fit(
                [self.geneDistTrain_dep, self.diseaseDistTrain_dep,
                    self.wordEmbedTrain_dep, self.depFeaturesTrain_dep],
                self.train_y_cat_dep, batch_size=self.batch_size, verbose=False,
                epochs=8, callbacks=[dep_csv_logger])
            probs = self.model_dep.predict(
                [self.geneDistTest_dep, self.diseaseDistTest_dep,
                 self.wordEmbedTest_dep, self.depFeaturesTest_dep],
                verbose=False)
            pred_test = np.argmax(probs, axis=1)

            # Calculate accuracy
            acc = np.sum(pred_test == self.yTest_dep) / \
                float(len(self.yTest_dep))
            self.max_acc_dep = max(self.max_acc_dep, acc)
            print "Accuracy: %.4f (max: %.4f)" % (acc, self.max_acc_dep)

            # Calculate F1 score
            f1Sum = 0
            f1Count = 0
            for targetLabel in [0, 1]:
                prec = self.getPrecision(pred_test, self.yTest_dep, targetLabel)
                rec = self.getPrecision(self.yTest_dep, pred_test, targetLabel)
                f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
                f1Sum += f1
                f1Count += 1

            macroF1 = f1Sum / float(f1Count)
            self.max_f1_dep = max(self.max_f1_dep, macroF1)
            print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, self.max_f1_dep)

        print "Start training: Without dependency"
        # Do the same with the model without dependency
        for epoch in xrange(self.nb_epoch):
            no_dep_csv_logger = CSVLogger('no_dep' + str(epoch) + '.log')
            self.model_no_dep.fit(
                [self.geneDistTrain_no_dep, self.diseaseDistTrain_no_dep,
                    self.wordEmbedTrain_no_dep],
                self.train_y_cat_no_dep, batch_size=self.batch_size, verbose=False,
                epochs=8, callbacks=[no_dep_csv_logger])
            probs = self.model_no_dep.predict(
                [self.geneDistTest_no_dep, self.diseaseDistTest_no_dep,
                 self.wordEmbedTest_no_dep], verbose=False)
            pred_test = np.argmax(probs, axis=1)

            acc = np.sum(pred_test == self.yTest_no_dep) / \
                float(len(self.yTest_no_dep))
            self.max_acc_no_dep = max(self.max_acc_no_dep, acc)
            print "Accuracy: %.4f (max: %.4f)" % (acc, self.max_acc_no_dep)

            f1Sum = 0
            f1Count = 0
            for targetLabel in [0, 1]:
                prec = self.getPrecision(
                    pred_test, self.yTest_no_dep, targetLabel)
                rec = self.getPrecision(
                    self.yTest_no_dep, pred_test, targetLabel)
                f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
                f1Sum += f1
                f1Count += 1

            macroF1 = f1Sum / float(f1Count)
            self.max_f1_no_dep = max(self.max_f1_no_dep, macroF1)
            print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, self.max_f1_no_dep)


def main():
    # Shows how this code is run
    rcnn = RelationCNN()
    rcnn.loadDataset()
    # Create 2 models
    rcnn.model_dep = rcnn.createModel(useDep=True)
    rcnn.model_no_dep = rcnn.createModel(useDep=False)
    rcnn.trainModel()
    # Save the models
    root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rcnn.model_dep.save(os.path.join(root_folder, "models/model_dep.h5"))
    rcnn.model_no_dep.save(os.path.join(root_folder, "models/model_no_dep.h5"))


if __name__ == "__main__":
    main()
