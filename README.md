# CS230: antibody language
Using language models to generate human antibodies.

__Authors:__ Mira Partha, Felipe Calero Forero, Aaron Behr


_Note: This is a work-in-progress; code is being updated and may be incomplete._

We have applied the [ESM deep Transformer language model](https://www.biorxiv.org/content/10.1101/622803v3) to the novel downstream task of generating candidate human antibodies. We have devised a method of evaluating these candidates and compare them to baselines.

The main procedure is executed in `workflow#main`:

- Load the ESM deep Transformer language model.
- Compute embeddings for the 89k candidate sequences with binding energy calculations, for use in training the downstream evaluation model.
- Generate a set of randomly-mutated candidate sequences to compare with the language model & compute embeddings for those candidates.
- (ToDo) Generate a set of candidate sequences mutated using a substitution matrix, to compare with the language model and compute their embeddings.
- Generate (ToDo: a set of) candidate sequences mutated using the language model. We plan to evaluate several ways of performing this "unmasking" task, which we discuss in the report. We expect that other strategies will be more effective than our current "naive" strategy.

(ToDo) The embeddings generated (1) randomly, (2) using a substitution matrix, and (3) by the language model will be compared using the regression model. 

Finally, we will attempt to augment and/or fine-tune the model to improve the performance on this task.


### Dependencies

```bash
# python>=3.6,<=3.8, miniconda may be easiest
pip3 install numpy torch torchvision torchaudio pandas xlrd
```
