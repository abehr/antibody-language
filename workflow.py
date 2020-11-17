import torch
import keras
import os
import numpy as np
import pandas as pd
import xlrd # for excel with pandas
import subprocess
import esm_src.esm as esm
import csv
from generators import Generators
import sequence_model_generators as model_gen

model_name = 'esm1_t34_670M_UR50S'
model_url = 'https://dl.fbaipublicfiles.com/fair-esm/models/%s.pt' % model_name
model_dir = 'models'
model_fp = os.path.join(model_dir, model_name + '.pt')
data_dir = 'data'
# cov1_ab_fp = os.path.join(data_dir, 'cov1-antibody.txt')
foldx_metadata_fp = os.path.join(data_dir, '89ksequences.xlsx')
seq89k_best100_fp = os.path.join(data_dir, 'best100.xlsx')

vocab = esm.constants.proteinseq_toks['toks']

embedding_dir = lambda name: os.path.join(data_dir, name + '_embeddings')
fasta_fp = lambda name: os.path.join(data_dir, name + '.fasta')

# 1-indexed list of indices to allowed to mutate
all_masks = [31,32,33,47,50,51,52,54,55,57,58,59,60,61,62,99,100,101,102,103,104,271,273,274,275,335,336,337,338,340,341]
all_fastas = ['seq85k', 'subset_seq89k', 'random_generated', 'substitution_generated', 'best100']

def run():
	print('Load initial SARS-CoV-1 antibody sequence')
	cov1_ab = load_cov1_template()
	compute_embeddings('cov1_antibody')

	# Compute embeddings for the 89k's best 100 to benchmark against model predictions
	load_and_convert_89k_best100()
	compute_embeddings('best100')

	# Load FoldX energy calculations for the 89k sequences
	import_energy_metadata()

	# Subset of the 89k seqs, for (test) training downstream model
	# compute_embeddings('subset_seq89k')
	# subset_seq89k_embeddings = load_seqs_and_embeddings('subset_seq89k', True, df)

	# Compute embeddings for the 85k sequences used in training the regression model
	compute_embeddings('seq85k')

	# Randomly generated mutations
	Generators.generate_random_predictions(cov1_ab, all_masks, 30)
	compute_embeddings('random_generated')

	# Substitution matrix-generated mutations
	Generators.generate_substitution_predictions(cov1_ab, all_masks, 30)
	compute_embeddings('substitution_generated')

	# Model-generated mutations (computes embeddings as well)
	model_gen.model_predict_seqs(model_gen.model_predict_seqs_1, cov1_ab, 10)
	model_gen.model_predict_seqs(model_gen.model_predict_seqs_2, cov1_ab, 10)
	model_gen.model_predict_seqs(model_gen.model_predict_seqs_3, cov1_ab, 10)
	model_gen.model_predict_seqs(model_gen.model_predict_seqs_4, cov1_ab, 10)


def compute_embeddings(name):
	print('Compute embeddings for %s' % name)
	assert os.path.exists(fasta_fp(name)), 'Fasta file for %s does not exist' % name
	assert not os.path.exists(embedding_dir(name)), 'Embeddings for %s already exist' % name

	# Download model manually, otherwise torch method from extract will put it in a cache
	if not os.path.exists(model_fp):
		print('Model does not exist locally - downloading %s' % model_name)
		if not os.path.exists(model_dir): os.mkdir(model_dir)
		subprocess.run(['curl', '-o', model_fp, model_url])

	# This script will automatically use GPU if possible, but will not have any errors if not. 
	subprocess.run(['python3', 'esm_src/extract.py', model_fp, fasta_fp(name), embedding_dir(name), 
		'--repr_layers', '34', '--include', 'mean', 'per_tok'])


def import_energy_metadata():
	# Get FoldX calculations from Excel spreadsheet
	assert os.path.exists(foldx_metadata_fp), 'FoldX data file %s does not exist' % foldx_metadata_fp
	print('Read FoldX data from Excel file')
	
	df = pd.read_excel(foldx_metadata_fp, sheet_name=1) # Sheet2

	# Output FoldX calculations (only) to CSV file for faster future import
	csv_fp = os.path.splitext(foldx_metadata_fp)[0] + '_foldx_only.csv'
	if not os.path.isfile(csv_fp):
		out_df = df[['Antibody_ID','FoldX_Average_Whole_Model_DDG', 'FoldX_Average_Interface_Only_DDG']]
		out_df.to_csv(csv_fp)

	return df


def import_energy_metadata_foldx():
	csv_fp = os.path.splitext(foldx_metadata_fp)[0] + '_foldx_only.csv'
	assert os.path.isfile(csv_fp), 'FoldX CSV file does not exist; you need to run import_energy_metadata() first'

	with open(csv_fp) as f:
		r = csv.reader(f)
		r.__next__() # skip header row
		d = {l[1]: np.array([l[2], l[3]]).astype('float32') for l in r}

	return d
	

def load_energy_metadata_foldx(seqs, foldx_dict):
	return np.stack([foldx_dict[seq] for seq in seqs])


# Use the full dataframe (which allows importing more info) rather than the faster FoldX-only.
# This function is mostly deprecated.
def load_energy_metadata(seqs, energy_metadata):
	metadata_dict = []
	for label in seqs:
		metadata = energy_metadata.loc[energy_metadata.Antibody_ID==label]
		assert metadata.shape[0] > 0, 'Expected a metadata entry for %s' % label
		metadata = metadata.iloc[0]
		metadata_dict.append([
			metadata.FoldX_Average_Whole_Model_DDG,
			metadata.FoldX_Average_Interface_Only_DDG
			# metadata.Statium
		])
	return np.stack(metadata_dict)


def load_and_convert_89k_best100():
	assert os.path.exists(seq89k_best100_fp), 'Data file %s dose not exist' % seq89k_best100_fp
	assert not os.path.exists(fasta_fp('best100')), '89k best 100 fasta file already exists'
	print('Read best 100 seqs (out of seq89k) from Excel file & write to fasta')

	df = pd.read_excel(seq89k_best100_fp)

	with open(fasta_fp('best100'), 'w') as f:
		for i in range(df.shape[0]):
			f.write('>%s\n' % df.iloc[i,0])
			f.write('%s\n' % df.iloc[i,1])


def get_embedding_list(name):
	assert os.path.exists(fasta_fp(name)), 'Fasta file for %s does not exist' % name
	assert os.path.exists(embedding_dir(name)), 'Embeddings for %s do not exist' % name
	return np.array([os.path.splitext(x)[0] for x in os.listdir(embedding_dir(name))])


def load_embeddings(name, batch, use_cpu=False):
	assert os.path.exists(embedding_dir(name)), 'Embeddings for %s do not exist' % name
	embeddings = []
	for seq in batch:
		f = os.path.join(embedding_dir(name), seq + '.pt')
		assert os.path.isfile(f), 'Requested embedding file(s) not found'
		if use_cpu or not torch.cuda.is_available():
			data = torch.load(f, map_location=torch.device('cpu'))
		else:
			data = torch.load(f)

		label = data['label']
		token_embeddings = np.delete(data['representations'][34], (0), axis=1)

		embeddings.append(torch.unsqueeze(token_embeddings, 0))

	X = torch.cat(embeddings, dim=0)
	X = torch.flatten(X, start_dim=1, end_dim=-1)
	X = X.numpy()
	X = keras.utils.normalize(X, axis=-1, order=2)
	return X


'''
energy_metadata expects a pandas dataframe (output of import_energy_metadata()).
	If provided, it adds FoldX calculations to the output.
subset is an optional list of sequence IDs. Normally, this function returns all
	of the embeddings found in the 'name' embedding dir. If provided, only return
	this subset. 

This function is only used for testing with small amounts of data, usually 
when running the workflow locally and using CPU. Mostly deprecated.
'''
def load_seqs_and_embeddings(name, use_cpu, energy_metadata=None, subset=None):
	print('Load seqs and embeddings for %s' % name)
	assert os.path.exists(fasta_fp(name)), 'Fasta file for %s does not exist' % name
	assert os.path.exists(embedding_dir(name)), 'Embeddings for %s do not exist' % name
	if energy_metadata is not None:
		assert type(energy_metadata) == pd.core.frame.DataFrame, 'Unexpected energy metadata type'


	print('Load embeddings from files and combine with metadata')
	embeddings_dict = {}
	for seq in (subset if subset else os.listdir(embedding_dir(name))):
		f = os.path.join(embedding_dir(name), seq + ('.pt' if subset else ''))
		assert os.path.isfile(f), 'Requested embedding file(s) not found'
		if use_cpu or not torch.cuda.is_available():
			data = torch.load(f, map_location=torch.device('cpu'))
		else:
			data = torch.load(f)

		label = data['label']
		token_embeddings = np.delete(data['representations'][34], (0), axis=1)
		# logits = np.delete(data['logits'], (0), axis=1)
		d = {'token_embeddings': token_embeddings}

		if energy_metadata is not None:
			metadata = energy_metadata.loc[energy_metadata.Antibody_ID==label]

			assert metadata.shape[0] > 0, 'Expected a metadata entry for %s' % label
			metadata = metadata.iloc[0] # ignore duplicate entries

			d['FoldX_Average_Whole_Model_DDG'] = metadata.FoldX_Average_Whole_Model_DDG
			d['FoldX_Average_Interface_Only_DDG'] = metadata.FoldX_Average_Interface_Only_DDG
			d['Statium'] = metadata.Statium

		embeddings_dict[label] = d

	return embeddings_dict


def load_cov1_template():
	with open(fasta_fp('cov1_antibody')) as f:
		f.readline() # skip label
		return f.readline().strip()





if __name__ == '__main__':
	run()