from typing import List
from random import random
import numpy as np

class generators():
    def random_gen(masked_template: str, num_seq: int, output_path: str, vocab : List[str] = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']) -> List[str]:
        random_seqs = set()
        #for n in range(num_seq):
        while len(random_seqs) < num_seq:
            mutant=''
            for residue in template:
                if residue == '?':
                    mutant += random.choice(vocab)
                else:
                    mutant += residue
            random_seqs.add(mutant)
        random_seqs = list(random_seqs)

        #Write sequences to .fasta file instead
        ofile = open(output_path, "w")               # e.g. 'directory/subdirectory/randomseqs.fasta'
        for i in range(len(random_seqs)):
            ofile.write(">random_seq" + str(i+1) + "\n" + random_seqs[i] + "\n")
        ofile.close()

        return random_seqs

    def substitution_gen(unmasked_template: str, num_seq: int, submat: np.array, vocab : List[str] = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']) -> List[str]:
        pass