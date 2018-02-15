import pickle
from random import randint
import sys

class DataGenerator():
    def __init__(self):
        self.sequences = []
        self.labels = []

    def generate_single_binary_sequence(self, sequence_length):
        sequence = []

        for i in range(sequence_length):
            sequence.append(randint(0, 1))

        return sequence

    def get_xor_parity_of_sequence(self, sequence):
        parity = sequence[0]
        
        for elem in sequence[1:]:
            parity ^= elem

        return parity

    def generate_fixed_length_sequences(self, sequence_length, num_sequences):

        self.sequences = []
        self.labels = []

        for i in range(num_sequences):
            generated_sequence = self.generate_single_binary_sequence(sequence_length)
            label = self.get_xor_parity_of_sequence(generated_sequence)
            self.sequences.append(generated_sequence)
            self.labels.append(label)

        return self.sequences, self.labels

    def generate_variable_length_sequences(self, min_sequence_length, max_sequence_length, num_sequences):

        self.sequences = []
        self.labels = []

        for i in range(num_sequences):
            sequence_length = randint(min_sequence_length, max_sequence_length)
            generated_sequence = self.generate_single_binary_sequence(sequence_length)
            label = self.get_xor_parity_of_sequence(generated_sequence)
            self.sequences.append(generated_sequence)
            self.labels.append(label)

        return self.sequences, self.labels

    def write_output_to_file(self, filepath):
        data = {'sequences': self.sequences, 'labels': self.labels}
        with open(filepath, 'wb') as fptr:
            pickle.dump(data, fptr)

def run(sysargs):
    dg = DataGenerator()

    output_file_path = sysargs[0]
    num_sequences = int(sysargs[1])
    min_sequence_length = int(sysargs[2])

    if len(sysargs) > 3 and sysargs[3] is not None and len(sysargs[3].strip()) > 0:
        max_sequence_length = int(sysargs[3])
        dg.generate_variable_length_sequences(min_sequence_length, max_sequence_length, num_sequences)

    else:
        dg.generate_fixed_length_sequences(min_sequence_length, num_sequences)

    dg.write_output_to_file(output_file_path)

if __name__ == '__main__':
    run(sys.argv[1:])
