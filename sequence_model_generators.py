import torch
import numpy as np
import subprocess
import esm_src.esm as esm
from argparse import Namespace
import random
import time
import os

data_dir = 'data'
model_fp = 'models/esm1_t34_670M_UR50S.pt'
all_masks = [31,32,33,47,50,51,52,54,55,57,58,59,60,61,62,99,100,101,102,103,104,271,273,274,275,335,336,337,338,340,341]
embedding_dir = lambda name: os.path.join(data_dir, name + '_embeddings')
fasta_fp = lambda name: os.path.join(data_dir, name + '.fasta')

def model_predict_seqs(prediction_method, initial_seq, num_iters, use_cpu=False):
	model, alphabet, batch_converter, initial_tokens = load_model_prediction_tools(initial_seq, use_cpu)

	predictions = []
	t0 = time.time()
	for i in range(num_iters):
		t1 = time.time()
		predictions.append(prediction_method(initial_tokens, model, alphabet, i+1))
		t2 = time.time()
		print('iter %d of %d: %.1f min (%.1f min total)' % (i+1, num_iters, (t2-t1)/60, (t2-t0)/60))

	# return compute_predicted_seq_embeddings(model, batch_converter, predictions)
	labels = [x[0] for x in predictions]
	tokens = np.delete(torch.cat([x[1] for x in predictions]), (0), axis=1) # remove BOS token
	strs = tokens2strs(alphabet, tokens)


	timestamp = time.strftime('%m%d_%H%M', time.localtime(time.time()))
	name = prediction_method.__name__ + '_' + timestamp

	# Write seqs to fasta file
	with open(fasta_fp(name), 'w') as f:
		for l,s in zip(labels, strs):
			f.write('>%s\n' % l)
			f.write('%s\n' % s)

	subprocess.run(['python3', 'esm_src/extract.py', model_fp, fasta_fp(name), embedding_dir(name), 
		'--repr_layers', '34', '--include', 'per_tok'])


def load_model_prediction_tools(seq, use_cpu):
	print('Load ESM model and convert template sequence')
	model, alphabet = load_local_model(use_cpu)
	batch_converter = alphabet.get_batch_converter()
		
	# Note this will also pad any sequence with different length
	labels, strs, initial_tokens = batch_converter([('cov1_ab', seq)])

	return model, alphabet, batch_converter, initial_tokens


def unmask_token(tokens, model, alphabet, idx=-1):
	with torch.no_grad():
		results = model(tokens, repr_layers=[34])
	tokens, _, logits = parse_model_results(tokens, results)
	softmax_predict_unmask(tokens, logits, idx)
	return tokens


# mask all 31 residues at once; unmask all at once.
# This is not expected to work well; only to be used for comparison
def model_predict_seqs_1(initial_tokens, model, alphabet, idx):
	name = 'M1_%d' % idx
	print(name)
	tokens = initial_tokens.detach().clone()
	apply_mask(tokens, all_masks) # mask all tokens
	tokens = unmask_token(tokens, model, alphabet) # unmask/predict all tokens
	return (name, tokens)


# mask/unmask one at a time, randomly, mu times (with replacement s.t. may or may not mutate all 31)
def model_predict_seqs_2(initial_tokens, model, alphabet, idx):
	mu = random.randint(1,31)
	name = 'M2_mu%d_%d' % (mu, idx)
	print(name)
	# TODO: should mu be constant
	tokens = initial_tokens.detach().clone()
	for i in range(mu):
		mask = all_masks[random.randint(0, len(all_masks)-1)] 
		print('Masking/unmasking single token at position %d (iter %d of %d)' % (mask, i+1, mu))
		apply_mask(tokens, [mask]) # mask a random token
		tokens = unmask_token(tokens, model, alphabet, mask) # unmask it using softmax dist prediction
		
	return (name, tokens)


# mask all 31 residues; unmask all one at a time in random order
def model_predict_seqs_3(initial_tokens, model, alphabet, idx):
	name = 'M3_%d' % idx
	print(name)
	tokens = initial_tokens.detach().clone()
	apply_mask(tokens, all_masks) # mask all tokens

	unmask_order = list(all_masks)
	random.shuffle(unmask_order)
	for i,mask in enumerate(unmask_order):
		print('Unmasking all tokens in random order (%d of 31)' % (i+1))
		tokens = unmask_token(tokens, model, alphabet, mask) # unmask all, one at a time, in random order

	return (name, tokens)



# mask a randomly-sized random subset of the 31 residues, unmask all one at a time in random order
def model_predict_seqs_4(initial_tokens, model, alphabet, idx):
	tokens = initial_tokens.detach().clone()
	
	# mask a randomly-sized random subset
	random_masks = list(all_masks)
	random.shuffle(random_masks)
	random_masks = random_masks[:random.randint(1, len(all_masks))]
	apply_mask(tokens, random_masks)
	

	name = 'M4_rand%d_%d' % (len(random_masks), idx)
	print(name)
	print('Masking %d (of 31) tokens at once' % len(random_masks))
	for i,mask in enumerate(random_masks):
		print('Unmasking %d of %d tokens' % (i+1, len(random_masks)))
		tokens = unmask_token(tokens, model, alphabet, mask) # unmask all, one at a time

	return (name, tokens)



def parse_model_results(batch_tokens, results, remove_bos_token=False):
	if remove_bos_token:
		tokens = np.delete(batch_tokens, (0), axis=1)
		token_embeddings = np.delete(results["representations"][34], (0), axis=1)
		logits = np.delete(results["logits"], (0), axis=1)
	else:
		tokens = batch_tokens
		token_embeddings = results["representations"][34]
		logits = results["logits"]

	return tokens, token_embeddings, logits


# tokens is 1-indexed because of BOS token; masks is also 1-indexed - no need to adjust anything.
def apply_mask(tokens, masks):
	for i in range(len(tokens)):
		for j in masks:
			tokens[i][j] = 33


def tokens2strs(alphabet, batch_tokens):
	return [''.join((alphabet.get_tok(t) for t in tokens)) for tokens in batch_tokens]


def load_local_model(use_cpu):
	# (tweaked from pretrained load model)
	alphabet = esm.Alphabet.from_dict(esm.constants.proteinseq_toks)
	model_data = torch.load(model_fp, map_location=torch.device('cpu'))

	pra = lambda s: ''.join(s.split('decoder_')[1:] if 'decoder' in s else s)
	prs = lambda s: ''.join(s.split('decoder.')[1:] if 'decoder' in s else s)
	model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
	model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}

	model = esm.ProteinBertModel(
	  Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx
	)
	model.load_state_dict(model_state)

	return model, alphabet


# Predict a specific token (predict_index) or predict all masked
def softmax_predict_unmask(batch_tokens, logits, predict_index=-1):
	sm = torch.nn.Softmax(dim=1)

	for i in range(len(batch_tokens)):
		if predict_index > -1:
			# convert to 2D tensor so it is the same dims as multiple masks
			softmax_masks = sm(logits[i][predict_index].view(1, 35))
		else:
			softmax_masks = sm(logits[i][batch_tokens[i] == 33])

		# torch.amax returns max
		if softmax_masks.size()[0] > 0:
			if predict_index > -1:
				batch_tokens[i][predict_index] = torch.multinomial(softmax_masks, 1)[0][0]
			else:
				batch_tokens[i][batch_tokens[i] == 33] = torch.multinomial(softmax_masks, 1)[:,0]