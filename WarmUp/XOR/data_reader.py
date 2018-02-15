import pickle
import numpy
import sys

class DataReader():
    def __init__(self):
        self.sequences = []
        self.labels = []

    def read_pkl_data_at_file_path(self, filepath):       
        with open(filepath, 'rb') as fptr:
            data = pickle.load(fptr)
        self.sequences = data['sequences']
        self.labels = data['labels']

    def get_sequences(self, convert_to_numpy_array = True):
        if convert_to_numpy_array:
            return numpy.array(self.sequences)
        return self.sequences

    def get_labels(self, convert_to_numpy_array = True):
        if convert_to_numpy_array:
            return numpy.array(self.labels)
        return self.labels


def run(sysargs):
    dr = DataReader()

    input_file_path = sysargs[0]
    dr.read_pkl_data_at_file_path(input_file_path)

    sequences = dr.get_sequences()
    labels = dr.get_labels()

    print('Average Length = ', sum([len(sequence) for sequence in sequences])/len(sequences))
    print('Minimum Length = ', min([len(sequence) for sequence in sequences]))
    print('Maximum Length = ', max([len(sequence) for sequence in sequences]))

    print('Number of odd parity = ', sum(1 for label in labels if label == 1))
    print('Number of even parity = ', sum(1 for label in labels if label == 0))

if __name__ == '__main__':
    run(sys.argv[1:])
