import os, sys, keras, pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from data_reader import DataReader
from keras import backend
from ast import literal_eval
from keras.models import model_from_json

TRAINED_MODEL_FILE = "trained_model.json"

class ModelFactory(object):
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'
    metrics=['accuracy']

    @staticmethod
    def create_lstm_model(embedding_input_dim=1, embedding_output_dim=256, lstm_hidden_cells_dim=128, dropout=0.5, activation_func='tanh'):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_input_dim+1, output_dim=embedding_output_dim))
        model.add(LSTM(lstm_hidden_cells_dim))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation=activation_func))
        model.compile(loss=ModelFactory.loss,
                    optimizer=ModelFactory.optimizer,
                    metrics=ModelFactory.metrics)

        return model

class TestBed(object):
    def __init__(self):
        self.model = None        
    
    def init_model(self, embedding_input_dim=1, embedding_output_dim=256, lstm_hidden_cells_dim=128, dropout=0.5, activation_func='tanh'):
        self.model = ModelFactory.create_lstm_model(embedding_input_dim, embedding_output_dim, lstm_hidden_cells_dim, dropout, activation_func)

    def train(self, train_data, labels, batch_size=16, epochs=1):
        print(self.__class__.__name__+":about to fit model...")
        self.model.fit(train_data, labels, batch_size=batch_size, epochs=epochs)
        print(self.__class__.__name__+":model training complete...")

    def save_model(self, model_file_name=TRAINED_MODEL_FILE):
        print("save_model:about to dump model to file.")
        model_json = self.model.to_json()
        with open(model_file_name, "w") as op:
            op.write(model_json)
        self.model.save_weights(model_file_name+".h5")
        print("save_model:model saved successfully.")

    def load_model(self, model_file_name=TRAINED_MODEL_FILE):
        json_file = open(model_file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_file_name+".h5")
        self.model.compile(optimizer=ModelFactory.optimizer, loss=ModelFactory.loss, metrics=ModelFactory.metrics)
        print("load_model:model loaded successfully from file: '" + model_file_name + "'")

    def test(self, test_data, labels, batch_size=16):        
        score = self.model.evaluate(test_data, labels, batch_size=16)
        metrics_names = self.model.metrics_names
        return (metrics_names, score)

    def predict(self, test_data, batch_size=16, verbose=True):
        self.model.predict(test_data, batch_size, verbose)


def run(sysargs):
    if len(sysargs) < 1:
        print("Insufficient input args.")
        print("Usage:")
        print("python lstm.py <input_file_path>")
    else:        
        skip_train_flag = False
        testbed = TestBed()
            
        if(len(sysargs)==2):
            skip_train_flag= literal_eval(sysargs[1])
        
        print("\nskip_train_flag:'" + str(skip_train_flag) + "'")
        
        input_file_path = sysargs[0]
        dr = DataReader()
        dr.read_pkl_data_at_file_path(input_file_path)
        sequences = dr.get_sequences()
        labels = dr.get_labels()

        if not skip_train_flag:            
            # train            
            testbed.init_model()
            testbed.train(sequences, labels)
            testbed.save_model()
        else:
            # skipping training part, load model
            testbed.load_model()
        
        metrics_names, score = testbed.test(sequences, labels)
        print("metrics_names:")
        print(metrics_names)
        print("score="+str(score))

        

if __name__ == "__main__":
    run(sys.argv[1:])
    backend.clear_session()
