import numpy as np
import os
from typing import List, Tuple, TypeVar, Generic

ArrayLike = TypeVar('ArrayLike')

class DataHandler:
    def __init__(self,
                 train,
                 test,
                 input_type,
                 region_size=30,
                 pool_size=40):

        self._NUCLEOTIDE_CONVERTER = {
            'A': 1,
            'C': 2,
            'G': 3,
            'T': 4,
            'N': 5,
        }

        self.train = train
        self.test = test
        self.input_type = input_type
        self.classes = [os.path.splitext(
            os.path.basename(e))[0] for e in self.test]
        self.num_classes = len(self.classes)
        self.max_len = 0
        self.vocabulary = ['A', 'C', 'G', 'T', 'N', 5]
        self.vocab_size = len(self.vocabulary)

        self._process_data()

    def _pad_sequences(self, num_seq_list: List[ArrayLike]) -> ArrayLike:
        return np.array([np.pad(seq,
                                (0, self.max_len - len(seq)),
                                'constant',
                                constant_values=(0, 0))
                         for seq in num_seq_list])

    def _process_data(self):
        x_train, y_train = [], []
        x_test, y_test = [], []

        if self.input_type == 'genomic':
            x_train, y_train = self._process_genomic_data(self.train)
            x_test, y_test = self._process_genomic_data(self.test)

        elif self.input_type == 'information-matrix':
            x_train, y_train = self._process_information_matrix_data(
                self.train)
            x_test, y_test = self._process_information_matrix_data(self.test)

        self.x_train = self._pad_sequences(x_train)
        self.y_train = y_train
        self.x_test = self._pad_sequences(x_test)
        self.y_test = y_test

        self.train_size = len(self.y_train)
        self.test_size = len(self.y_test)

    def _get_class_from_path(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _convert_nucleotide(self, nucleotide: str) -> int:
        if nucleotide in self._NUCLEOTIDE_CONVERTER:
            return self._NUCLEOTIDE_CONVERTER[nucleotide]
        else:
            return 5

    def _convert_sequence(self, seq: str) -> ArrayLike:

        num_seq = np.array([self._convert_nucleotide(c) for c in seq],
                           dtype=np.uint8)

        seq_len = len(seq)
        if seq_len > self.max_len:
            self.max_len = seq_len

        return num_seq

    def _process_genomic_data(self,
                              files: List[str]
                              ) -> Tuple[ArrayLike, ArrayLike]:
        seqs, labels = [], []

        for fl in files:
            with open(fl, 'r') as f:
                class_index = self.classes.index(self
                                                 ._get_class_from_path(fl))

                for line in f.readlines():
                    if line[0] == '>':
                        if seqs != []:
                            seqs[-1] = self._convert_sequence(seqs[-1])
                        seqs.append('')
                        labels.append(class_index)
                    else:
                        seqs[-1] += line.upper().strip()

                seqs[-1] = self._convert_sequence(seqs[-1])

        return [e for e in seqs], np.array([e for e in labels])

    def _process_information_matrix_data(self,
                                         files: List[str]
                                         ) -> Tuple[ArrayLike, ArrayLike]:
        # TODO implement the information matrix processing
        return np.array([]), np.array([])
