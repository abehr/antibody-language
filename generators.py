from typing import List
# import random
import numpy as np
import os

data_dir = 'data'
fasta_fp = lambda name: os.path.join(data_dir, name + '.fasta')
import esm_src.esm as esm
vocab = esm.constants.proteinseq_toks['toks']

# 1-indexed list of indices to allowed to mutate
masks = [31,32,33,47,50,51,52,54,55,57,58,59,60,61,62,99,100,101,102,103,104,271,273,274,275,335,336,337,338,340,341]
seq = 'QVQLQQSGAEVKKPGSSVKVSCKASGGTFSSYTISWVRQAPGQGLEWMGGITPILGIANYAQKFQGRVTITTDESTSTAYMELSSLRSEDTAVYYCARDTVMGGMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTSALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTSPLFVHHHHHHGDYKDDDDKGSYELTQPPSVSVAPGKTARITCGGNNIGSKSVHWYQQKPGQAPVLVVYDDSDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDSSSDYVFGTGTKVTVLGQPKANPTVTLFPPSSEEFQANKATLVCLISDFYPGAVTVAWKADGSPVKAGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS'
# num_seqs = 100

sub_mat_fp = os.path.join(data_dir, 'blosum80.txt')

class generators():
	def generate_random_predictions(seq, masks, num_seqs):
		name = 'random_generated'
		print('Generate %s predictions' % name)
		assert not os.path.exists(fasta_fp(name)), '%s fasta already exists' % name
	
		random_seqs = set()
		while len(random_seqs) < num_seqs:
			mutant=''
			for i,residue in enumerate(seq):
				if (i+1) in masks:
					mutant += np.random.choice(vocab)
				else:
					mutant += residue
			random_seqs.add(mutant)
		random_seqs = list(random_seqs)

		with open(fasta_fp(name), 'w') as f:
			for n in range(num_seqs):
				f.write('>%s_%d\n' % (name, n+1))
				f.write('%s\n' % random_seqs[n])


	def generate_substitution_predictions(seq, masks, num_seqs):
		name = 'substitution_generated'
		print('Generate %s predictions' % name)
		assert not os.path.exists(fasta_fp(name)), '%s fasta already exists' % name

		#Load in substitution matrix
		submat = np.loadtxt(sub_mat_fp, delimiter='\t', skiprows=2)
		scale = np.log(2.0)/2.0  # Lambda scaling factor for BLOSUM80

		amino_acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
		#Prior probabilities of amino acids
		priors=[0.06, 0.107, 0.042, 0.036, 0.026, 0.047, 0.039, 0.072, 0.030, 0.052, 0.079, 0.055, 0.018, 0.022, 0.050, 0.087, 0.069, 0.016, 0.031, 0.061]
		
		probmat = np.zeros((20,20))
		for i in range(20):
			for j in range(20):
				probmat[i][j] = priors[i]*priors[j]*2**(scale*submat[i][j])

		sub_seqs = set()
		while len(sub_seqs) < num_seqs:
			mutant=''
			for i,residue in enumerate(seq):
				if (i+1) in masks:
					dist=probmat[amino_acids.index(residue),:]
					print(np.sum(dist))
					mutant += np.random.choice(amino_acids, p=dist)
				else:
					mutant += residue
			sub_seqs.add(mutant)
		sub_seqs = list(sub_seqs)

		with open(fasta_fp(name), 'w') as f:
			for n in range(num_seqs):
				f.write('>%s_%d\n' % (name, n+1))
				f.write('%s\n' % sub_seqs[n])