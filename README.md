# CS230: Antibody Language
Generating human antibodies using language models

__Authors:__ Mira Partha, Felipe Calero Forero, Aaron Behr


We have applied the [ESM deep Transformer language model](https://www.biorxiv.org/content/10.1101/622803v3) to the novel downstream task of generating candidate human antibodies. We have devised a method of evaluating these candidates and compare them to baselines.


### Outline of the pipeline

- Compute embeddings for the 89k candidate sequences with binding energy calculations, for use in training the downstream evaluation model.
- Generate a set of randomly-mutated candidate sequences to compare with the language model & compute embeddings for those candidates.
- Generate a set of candidate sequences mutated using a substitution matrix, to compare with the language model and compute their embeddings.
- Generate a set of candidate sequences mutated using the language model. We evaluate several methods of performing this multi-token prediction task, which we discuss in the report.
- Train a regression model to predict FoldX binding energy from per-token sequence embedding.
- Finally, compare the embeddings generated (1) randomly, (2) using a substitution matrix, and (3) by the language model using the regression model.


The full pipeline takes around a day (24h) to run on a p2.xlarge AMI, with around 2 TB of disk space.


### Running the pipeline

```bash
# Python 3.7 recommended (3.8 or 3.6 may also work)

# Install dependencies
pip install numpy torch torchvision torchaudio keras pandas xlrd

# Load language model, compute embeddings, generate predicted sequences
python workflow.py

# Train binding energy prediction model, evaluate predicted sequences
python binding_energy_model.py # prints metrics to compare sequence generation methods
```