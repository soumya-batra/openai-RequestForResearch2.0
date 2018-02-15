import os, sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

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
        self.init_model()


    def init_model(self):
        self.model = ModelFactory.create_lstm_model()

    def train(train_data, labels, batch_size=16, epochs=10):
        self.model.fit(train_data, labels, batch_size=batch_size, epochs=epochs)

    def predict(test_data, labels, batch_size=16):
        score = self.model.evaluate(test_data, labels, batch_size=16)


def run(sysargs):
    if len(sysargs) < 1:
        print(":(")
    else:
        input_filename = sysargs[0]
        

if __name__ == "__main__":
    run(sys.argv[1:])
