import tensorflow as tf
from tensorflow.contrib import rnn


class WordRNN(object):
    def __init__(self, vocabulary_size, max_document_length, num_class, hidden_layer_num=4, embedding_size=256,
                 num_hidden=100, fc_num_hidden=256):
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.fc_num_hidden = fc_num_hidden
        self.hidden_layer_num = hidden_layer_num

        self.x = tf.placeholder(tf.int32, [None, max_document_length])
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.x)[0]

        with tf.variable_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            x_emb = tf.nn.embedding_lookup(embeddings, self.x)
        with tf.variable_scope("rnn"):
            def lstm_cell():
                return rnn.BasicLSTMCell(self.num_hidden)

            cell = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell() for _ in range(self.hidden_layer_num)])  # , state_is_tuple=True)
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x_emb, initial_state=initial_state, sequence_length=self.x_len,
                                               dtype=tf.float32)
            rnn_output_flat = tf.reshape(rnn_outputs, [-1, max_document_length * self.num_hidden])

        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(rnn_output_flat, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(dropout, num_class)
            # self.logits = tf.layers.dense(dropout, num_class, activation=tf.nn.relu)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
