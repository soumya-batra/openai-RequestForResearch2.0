import os, sys, keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from data_reader import DataReader

class ModelFactory(object):
    @staticmethod
    def create_lstm_model(embedding_input_dim=1, embedding_output_dim=256, lstm_hidden_cells_dim=128, dropout=0.5, activation_func='tanh'):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_input_dim, output_dim=embedding_output_dim))
        model.add(LSTM(lstm_hidden_cells_dim))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation=activation_func))
        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        return model

class TestBed(object):
    def __init__(self):
        self.model = None
    
    def init_model(self, embedding_input_dim=1, embedding_output_dim=256, lstm_hidden_cells_dim=128, dropout=0.5, activation_func='tanh'):
        self.model = ModelFactory.create_lstm_model(embedding_input_dim, embedding_output_dim, lstm_hidden_cells_dim, dropout, activation_func)

    def train(self, train_data, labels, batch_size=16, epochs=10):
        print(self.__class__.__name__+":about to fit model...")
        self.model.fit(train_data, labels, batch_size=batch_size, epochs=epochs)
        print(self.__class__.__name__+":model training complete...")

    def test(self, test_data, labels, batch_size=16):
        score = self.model.evaluate(test_data, labels, batch_size=16)

    def predict(self, test_data, batch_size=16, verbose=True):
        self.model.predict(test_data, batch_size, verbose)


def run(sysargs):
    if len(sysargs) < 1:
        print(":(")
    else:
        input_file_path = sysargs[0]
        dr = DataReader()
        dr.read_pkl_data_at_file_path(input_file_path)

        sequences = dr.get_sequences()
        labels = dr.get_labels()

        # train
        testbed = TestBed()
        testbed.init_model()
        testbed.train(sequences, labels)

        # test
        score = testbed.test(sequences, labels)
        print("score="+str(score))

        

if __name__ == "__main__":
    run(sys.argv[1:])
