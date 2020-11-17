import time
import os
import numpy as np
import torch
import torch.nn as nn
import keras
from keras.models import Sequential
from keras.layers import Dense
import workflow
from embedding_generator import EmbeddingGenerator

name = 'seq85k'
use_cpu = False

foldx_dict = workflow.import_energy_metadata_foldx()
seqs = workflow.get_embedding_list(name)
np.random.shuffle(seqs) # ensure random order

input_shape = 585782

dropout = .2


def main():
    print('AbReg - Initialize model')
    model = RegressionModel()

    print('AbReg - Compile model')
    model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())


    batch_size = 256 # 16-1024

    train_data = EmbeddingGenerator(name, seqs[:80000], foldx_dict, batch_size)
    valid_data = EmbeddingGenerator(name, seqs[80000:81000], foldx_dict, batch_size)
    test_data = EmbeddingGenerator(name, seqs[81000:82000], foldx_dict, batch_size)

    print('AbReg - Train model')
    model.fit(train_data, validation_data=valid_data, epochs=1)

    timestamp = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    model_fp = os.path.join('models', 'abreg_' + timestamp)

    print('AbReg - Save model to file %s' % model_fp)
    model.save(model_fp)

    model.trainable = False

    print('AbReg - Evaluate model')
    model.evaluate(test_data)
    # Use fit rather than evaluate (with frozen params) to get around the known issue
    # with keras BatchNorm
    model.fit(test_data) 

    evaluate(model)


def RegressionModel():
	X_input = keras.Input(shape=input_shape)

	X = keras.layers.Dropout(dropout)(X_input)

	for _ in range(8):
		X = Dense(800, activation='relu', kernel_initializer="he_uniform")(X)
		X = keras.layers.LayerNormalization(axis=-1)(X)
		X = keras.layers.Dropout(dropout)(X)

	X = Dense(700, activation='relu', kernel_initializer="he_uniform")(X)
	X = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(X)
	X = keras.layers.Dropout(dropout)(X)

	X = Dense(400, activation='relu', kernel_initializer="he_uniform")(X)
	X = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(X)
	X = keras.layers.Dropout(dropout)(X)

	X = Dense(2, kernel_initializer="he_uniform")(X)

	model = keras.Model(inputs = X_input, outputs = X, name='RegressionModel')

	return model
    

def evaluate(model):
	predictions = {}
	prediction_types = [
		'cov1_antibody',
		'best100',
		'random_generated',
		'substitution_generated',
		'model_predict_seqs_1_1117_0951',
		'model_predict_seqs_2_1117_0632',
		'model_predict_seqs_3_1117_0703',
		'model_predict_seqs_4_1117_0721'
	]

	for n in prediction_types:
		print('AbReg - Predict generated embeddings: ' + n)
		labels = workflow.get_embedding_list(n)
		# Note: batch size *must* not be larger than data size
		predicted_embeddings = EmbeddingGenerator(n, labels, foldx_dict, len(labels), include_targets=False)
		predicted_energies = model.predict(predicted_embeddings)
		predictions[n] = predicted_energies

	print('AbReg - Evaluate predicted whole-model binding energy of generated embeddings')
	for n,e in predictions.items():
		print('\nPredicted energies for ' + n)
		print('Mean: ' + str(np.mean(e, axis=0)[0]))
		print('Stdev: ' + str(np.sqrt(np.var(e, axis=0))[0]))

		el1 = list(e)
		el1.sort(key = lambda x: x[0])
		print('Best whole-model FoldX: ' + str(el1[0][0]))
		print('Worst whole-model FoldX: ' + str(el1[-1][0]))


def load_from_file(fp):
	model = keras.models.load_model(fp)
	model.trainable = False
	return model



if __name__ == '__main__':
	main()