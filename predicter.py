import os
import pickle
import numpy as np
import tensorflow as tf
from keras.engine.saving import load_model

FOLDER = "mk5_2"


def main(_):
    with open(os.path.join(FOLDER, "chars.txt"), "r", encoding="utf-8") as char_file:
        chars = char_file.read().splitlines()
        char_file.close()
    chars.insert(0, "\n")
    char_indexes = pickle.load(open(os.path.join(FOLDER, "char_indexes.pkl"), "rb"))
    index_char = pickle.load(open(os.path.join(FOLDER, "index_char.pkl"), "rb"))
    model = load_model(os.path.join(FOLDER, "model.h5"))

    text = input("Write the text : ")
    text += " "
    for x in range(500):
        text_matrix = np.zeros((1, len(text), len(chars)))
        for _x, _c in enumerate(text):
            text_matrix[0, _x, char_indexes[_c]] = 1
        prob = model.predict(text_matrix)[0]
        prob = np.asarray(prob).astype("float64")
        prob = np.log(prob) / 0.6
        exp_prob = np.exp(prob)
        pred = exp_prob / np.sum(exp_prob)
        sol_list = np.random.multinomial(1, pred, 1)

        _char = index_char[np.argmax(sol_list)]
        text = text[:] + _char

    print(text)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    tf.compat.v1.app.run(main=None, argv=None)
