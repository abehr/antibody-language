import workflow
import keras
import numpy as np

class EmbeddingGenerator(keras.utils.Sequence):
    def __init__(self, name, labels, energy_data, batch_size, include_targets=True, use_cpu=False):
        self.name = name
        self.labels = labels
        self.foldx_dict = energy_data
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.use_cpu = use_cpu
        self.indexes = np.arange(len(self.labels))

    def __len__(self):
        # number of batches per epoch
        return len(self.labels) // self.batch_size

    def __getitem__(self, idx):
        # retrieve batch of index
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]
        labels = self.labels[indexes]

        inputs = workflow.load_embeddings(self.name, labels, self.use_cpu)

        if self.include_targets:
            outputs = workflow.load_energy_metadata_foldx(labels, self.foldx_dict)
            return inputs, outputs
        else:
            return inputs
