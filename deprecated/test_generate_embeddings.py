

import torch
import os
import esm # pip3 install --user git+https://github.com/facebookresearch/esm.git
from argparse import Namespace

model_name = 'esm1_t34_670M_UR50S'
model_url = 'https://dl.fbaipublicfiles.com/fair-esm/models/%s.pt' % model_name
model_dir = 'models'
model_path = '%s/%s.pt' % (model_dir, model_name)
fasta_path = '../esm/examples/P62593.fasta'
embedding_dir = 'P62593_reprs'

# Download model manually, otherwise torch method from extract will put it in a cache
if not os.path.exists(model_path):
	if not os.path.exists(model_dir): os.mkdir(model_dir)
	subprocess.run(['curl', '-o', model_path, model_url])


model, alphabet = load_local_model(model_path, use_cpu=True)
batch_converter = alphabet.get_batch_converter()


data = [
	('prot1', 'MKK'), # mask it -> it thinks the best thing is K
	('prot2', 'MKK') # see diff
	('prot3', 'MYK')
]


data = [
	('md5_3ea41ceafb0cd3412378236fdf65149c','QVQLQQSGAEVKKPGSSVKVSCKASGGTFSDYTISWVRQAPGQGLELMGYITPILGMANYAQKFQGRVTITTDESTSTAYMELSSLRSEDTAVYYCARVTQMGGMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTSALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTSPLFVHHHHHHGDYKDDDDKGSYELTQPPSVSVAPGKTARITCGGNNIGRKSVHWYQQKPGQAPVLVVYDDSDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDMWSDYVFGTGTKVTVLGQPKANPTVTLFPPSSEEFQANKATLVCLISDFYPGAVTVAWKADGSPVKAGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS'),
	('md5_af0d944a9387a0b04b9c66677ae24807','QVQLQQSGAEVKKPGSSVKVSCKASGGTFSMYTISWVRQAPGQGLEWMGGITPILGIALYAQKFQGRVTITTDESTSTAYMELSSLRSEDTAVYYCARDTRMGGMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTSALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTSPLFVHHHHHHGDYKDDDDKGSYELTQPPSVSVAPGKTARITCGGNNIGSKSVHWYQQKPGQAPVLVVYDDSDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDFWSDYVFGTGTKVTVLGQPKANPTVTLFPPSSEEFQANKATLVCLISDFYPGAVTVAWKADGSPVKAGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS')
]

# Prepare data (two Ab seqs)
labels, strs, tokens = batch_converter(data)

# tokens[1][2] = 33

# Extract per-residue embeddings (on CPU)
# results is a dict with 'logits' and 'representations'
# When you call model(), it automatically runs forward() (extends torch.nn.Module)
with torch.no_grad():
    results = model(tokens, repr_layers=[34])


# Note that token 0 for each seq is BOS so it should be ignored. First residue is token 1.
token_embeddings = np.delete(results["representations"][34], (0), axis=1) # go from (2, 459, 1280) -> (2, 458, 1280)
logits = np.delete(results["logits"], (0), axis=1)
tokens = np.delete(tokens, (0), axis=1)

'''
# OPTIONALLY, Generate per-sequence embeddings via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# average over all tokens (per embed feat) to get the sequence embedding
sequence_embeddings = []
for i, (_, seq) in enumerate(data):
    sequence_embeddings.append(token_embeddings[i, 1:len(seq) + 1].mean(0))
'''



def load_local_model(model_path, use_cpu):
	# (tweaked from pretrained load model)
	alphabet = esm.Alphabet.from_dict(esm.constants.proteinseq_toks)
	model_data = torch.load(model_path, map_location=torch.device('cpu'))

	pra = lambda s: ''.join(s.split('decoder_')[1:] if 'decoder' in s else s)
	prs = lambda s: ''.join(s.split('decoder.')[1:] if 'decoder' in s else s)
	model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
	model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}

	model = esm.ProteinBertModel(
	  Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx
	)
	model.load_state_dict(model_state)

	return model, alphabet
