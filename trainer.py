import os
import re
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

flags = tf.compat.v1.flags
tf.compat.v1.flags.DEFINE_integer("sq_length",
                                  50,
                                  "Sequence length")
tf.compat.v1.flags.DEFINE_integer("step_size",
                                  3,
                                  "Step size")
tf.compat.v1.flags.DEFINE_string("input_folder",
                                 "texts",
                                 "Input folder")
tf.compat.v1.flags.DEFINE_string("output_folder",
                                 "mk6",
                                 "Output location")

tf.compat.v1.flags.DEFINE_integer("batch_size",
                                  512,
                                  "Model batch size")
tf.compat.v1.flags.DEFINE_integer("epochs",
                                  50,
                                  "Model epoch")
FLAGS = flags.FLAGS


def _preprocess(text):
    text = re.sub("[^a-zA-Z0-9ığüşöçİĞÜŞÖÇ \n\.\?\!\-\,\(\)\;\:\'\"]", " ", text)
    return text


def _create_map(text):
    _char = sorted(list(set(text)))
    _char_indexes = {}
    _index_char = {}

    for _x in range(len(_char)):
        _char_indexes[_char[_x]] = _x
        _index_char[_x] = _char[_x]
    return _char, _char_indexes, _index_char


def _split(text):
    _list = []
    _next = []
    for x in range(0, len(text) - FLAGS.sq_length, FLAGS.step_size):
        _list.append(text[x:x + FLAGS.sq_length])
        _next.append(text[x + FLAGS.sq_length])
    return _list, _next


def main(_):
    assert FLAGS.sq_length, " sequence length is missing..."
    assert FLAGS.step_size, " step size is missing..."
    assert FLAGS.input_folder, " input folder is missing..."
    assert FLAGS.output_folder, " output folder is missing..."

    text = ""
    for _file_name in os.listdir(FLAGS.input_folder):
        _file = open(os.path.join(FLAGS.input_folder, _file_name), "r", encoding="utf-8")
        _file_text = _file.read()
        _file_text = _preprocess(_file_text)
        text += _file_text

    _chars, _char_indexes, _index_char = _create_map(text)
    _sequence, _next = _split(text)

    X = np.zeros((len(_sequence), FLAGS.sq_length, len(_chars)), dtype=np.bool)
    Y = np.zeros((len(_sequence), len(_chars)), dtype=np.bool)
    for _x, _s in enumerate(_sequence):
        for _y, _char in enumerate(_s):
            X[_x, _y, _char_indexes[_char]] = 1
        Y[_x, _char_indexes[_next[_x]]] = 1

    model = create_model(len(_chars))
    for x in range(FLAGS.epochs):
        model.fit(x=X, y=Y, batch_size=FLAGS.batch_size, epochs=1)
        save_model(model, _chars, _char_indexes, _index_char, x)


def create_model(input_length):
    _model = Sequential()
    _model.add(LSTM(256, input_shape=(None, input_length), return_sequences=True))
    _model.add(LSTM(256))

    _model.add(Dropout(0.4))
    _model.add(Dense(256, activation="relu"))
    _model.add(Dropout(0.2))

    _model.add(Dense(input_length, activation="softmax"))
    _model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.005),
        metrics=["accuracy"]
    )
    return _model


def save_model(model, chars, char_indexes, index_char, index):
    output_folder = FLAGS.output_folder + "_" + str(index)
    os.mkdir(output_folder)

    model.save(os.path.join(output_folder, "model.h5"))

    _char_file = open(os.path.join(output_folder, "chars.txt"), "w", encoding="utf-8")
    for _c in chars[1:]:
        _char_file.write(_c + "\n")
    _char_file.close()

    _for_char_indexes = open(os.path.join(output_folder, "char_indexes.pkl"), "wb")
    pickle.dump(char_indexes, _for_char_indexes)
    _for_char_indexes.close()

    _for_index_char = open(os.path.join(output_folder, "index_char.pkl"), "wb")
    pickle.dump(index_char, _for_index_char)
    _for_index_char.close()


def plot_data(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    tf.compat.v1.app.run(main=None, argv=None)
