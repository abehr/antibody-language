from random import random
import numpy as np

class generators():
    def random_gen(masked_template: str, num_seq: int, vocab=['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']: List[str]) -> List[str]:
        random_seqs = []
        for n in range(num_seq):
            mutant=''
            for residue in template:
                if residue == '?':
                    mutant += random.choice(vocab)
                else:
                    mutant += residue
            random_seqs.append(mutant)
        return random_seqs

    def substitution_gen(unmasked_template: str, num_seq: int, submat: np.array, vocab=['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']: List[str]) -> List[str]:
        pass