# -*- coding: utf-8 -*-
"""AntibodyBindingEnergyRegressionModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N2t4JCxG1uFA__bM3Hu4RoD6_Dvhn_sP
"""

import numpy as np
import torch
import torch.nn as nn
import keras
from keras.models import Sequential
from keras.layers import Dense
import workflow


# subset_200_seq85k_embeddings = load_seqs_and_embeddings('subset_200_seq85k', False, df)

# subset = subset_200_seq85k_embeddings

# embeddings = []
# scores = []

# for key in subset.keys():
#   embedding = torch.unsqueeze(subset[key]['token_embeddings'], 0)
#   embeddings.append(embedding)
#   score = np.zeros(3)
#   score[0] = subset[key]['FoldX_Average_Whole_Model_DDG']
#   score[1] = subset[key]['FoldX_Average_Interface_Only_DDG']
#   score[2] = subset[key]['Statium']
#   scores.append(score)
# X = torch.cat(embeddings, dim=0)
# X = torch.flatten(X, start_dim=1, end_dim=-1)
# X = X.numpy()

# Y = np.stack(scores)

# X = keras.utils.normalize(X, axis=-1, order=2)
name = 'subset_seq85k'

df = workflow.import_energy_metadata()
X_train = workflow.get_embedding_list(name)
Y = workflow.load_energy_metadata(X_train, df)
Y_train = Y[:, 0:1]

input_shape = 1

dropout = .2

def RegressionModel(input_shape, dropout=.2):
    X_input = keras.Input(input_shape)

    X = keras.layers.Lambda(lambda batch: workflow.load_embeddings(name, batch), dynamic=True)(X_input)
    X = keras.layers.Dropout(dropout)(X)


    X = Dense(800, activation='relu', kernel_initializer="he_uniform")(X)
    X = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(X)
    X = keras.layers.Dropout(dropout)(X)


    X = Dense(700, activation='relu', kernel_initializer="he_uniform")(X)
    X = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(X)
    X = keras.layers.Dropout(dropout)(X)


    X = Dense(400, activation='relu', kernel_initializer="he_uniform")(X)
    X = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(X)
    X = keras.layers.Dropout(dropout)(X)


    X = Dense(3, kernel_initializer="he_uniform")(X)

    model = keras.Model(inputs = X_input, outputs = X, name='RegressionModel')

    return model

model = RegressionModel(input_shape)

model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

model.fit(x=X_train, y=Y_train, batch_size=2, epochs=1)