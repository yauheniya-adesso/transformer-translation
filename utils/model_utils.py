import numpy as np
import tensorflow as tf

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:,0::2] = np.sin(angle_rads[:,0::2])
    pos_encoding[:,1::2] = np.cos(angle_rads[:,1::2])
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)