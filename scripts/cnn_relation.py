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


batch_size = 64
n_filters = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 50

print "Load dataset"
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(os.path.join(root_folder, "data/round2/preprocessed.p"), "r") as f:
    yTrain, wordEmbedTrain, geneDistTrain, diseaseDistTrain, wordDistTrain, treeDistTrain, \
        geneTagTrain, diseaseTagTrain, lcsTagTrain, lcsPosTrain, lcsTextTrain \
        = pickle.load(f)
    yTest, wordEmbedTest, geneDistTest, diseaseDistTest, wordDistTrain, treeDistTest, \
        geneTagTest, diseaseTagTest, lcsTagTest, lcsPosTest, lcsTextTest \
        = pickle.load(f)
#
max_position = max(np.max(geneDistTrain), np.max(diseaseDistTrain)) + 1
#
n_out = max(yTrain) + 1
train_y_cat = np_utils.to_categorical(yTrain, n_out)

print "wordEmbedTrain: ", wordEmbedTrain.shape
print "geneDistTrain: ", geneDistTrain.shape
print "diseaseDistTrain: ", diseaseDistTrain.shape
print "yTrain: ", yTrain.shape

print "wordEmbedTest: ", wordEmbedTest.shape
print "geneDistTest: ", geneDistTest.shape
print "diseaseDistTest: ", diseaseDistTest.shape
print "yTest: ", yTest.shape

# f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
# embeddings = pkl.load(f)
# f.close()
#
# print "Embeddings: ", embeddings.shape
#
geneInput = Input(shape=(geneDistTrain.shape[1],))
geneEmbed = Embedding(input_dim=max_position, output_dim=position_dims,
                      input_length=geneDistTrain.shape[1])(geneInput)

diseaseInput = Input(shape=(diseaseDistTrain.shape[1],))
diseaseEmbed = Embedding(input_dim=max_position, output_dim=position_dims,
                         input_length=diseaseDistTrain.shape[1])(diseaseInput)
# TODO: word embeddings
# wordEmbedModel = Sequential()
# wordEmbedModel.add(Embedding(embeddings.shape[0], embeddings.shape[1],
#                              input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False))

mergedEmbed = concatenate([geneEmbed, diseaseEmbed])

convolution = Conv1D(filters=n_filters,
                     kernel_size=filter_length,
                     padding='same',
                     activation='tanh')(mergedEmbed)
# we use standard max over time pooling
max_pool = GlobalMaxPooling1D()(convolution)
#
dropout = Dropout(0.25)(max_pool)
dense_out = Dense(n_out, activation='softmax')(dropout)

model = Model(inputs=[geneInput, diseaseInput], outputs=[dense_out])
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])
model.summary()
print "Start training"
max_prec, max_rec, max_acc, max_f1 = 0, 0, 0, 0


def getPrecision(pred_test, yTest, targetLabel):
    # Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0

    for idx in xrange(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1

            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1

    if correctTargetLabelCount == 0:
        return 0

    return float(correctTargetLabelCount) / targetLabelCount


for epoch in xrange(nb_epoch):
    # model.fit([wordEmbedTrain, geneDistTrain, diseaseDistTrain],
    model.fit([geneDistTrain, diseaseDistTrain],
              train_y_cat, batch_size=batch_size, verbose=True, epochs=1)
    probs = model.predict(
        [geneDistTest, diseaseDistTest], verbose=True)
    pred_test = np.argmax(probs, axis=1)

    dctLabels = np.sum(pred_test)
    print dctLabels
    totalDCTLabels = np.sum(yTest)
    print totalDCTLabels

    acc = np.sum(pred_test == yTest) / float(len(yTest))
    max_acc = max(max_acc, acc)
    print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)

    f1Sum = 0
    f1Count = 0
    for targetLabel in xrange(1, max(yTest)):
        prec = getPrecision(pred_test, yTest, targetLabel)
        rec = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1Sum += f1
        f1Count += 1

    macroF1 = f1Sum / float(f1Count)
    max_f1 = max(max_f1, macroF1)
    print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1)
