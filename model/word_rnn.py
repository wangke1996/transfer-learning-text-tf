import tensorflow as tf
from tensorflow.contrib import rnn


class WordRNN(object):
    def __init__(self, vocabulary_size, max_document_length, num_class, hidden_layer_num=3, embedding_size=256,
                 num_hidden=200, fc_num_hidden=256, bi_direction=False, hidden_layer_num_bi=2, num_hidden_bi=100):
        self.embedding_size = embedding_size
        self.bi_direction = bi_direction
        if self.bi_direction:
            self.num_hidden = num_hidden_bi
            self.hidden_layer_num = hidden_layer_num_bi
        else:
            self.num_hidden = num_hidden
            self.hidden_layer_num = hidden_layer_num
        self.fc_num_hidden = fc_num_hidden

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

            if not self.bi_direction:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(self.hidden_layer_num)])  # , state_is_tuple=True)
                initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
                rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x_emb, initial_state=initial_state, sequence_length=self.x_len,
                                                   dtype=tf.float32)
                rnn_output_flat = tf.reshape(rnn_outputs, [-1, max_document_length * self.num_hidden])
            else:
                cell_fw = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(self.hidden_layer_num)])  # , state_is_tuple=True)
                cell_bw = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(self.hidden_layer_num)])  # , state_is_tuple=True)
                initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
                initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_emb,
                                                                 initial_state_fw=initial_state_fw,
                                                                 initial_state_bw=initial_state_bw,
                                                                 sequence_length=self.x_len, dtype=tf.float32)
                rnn_output_flat = tf.reshape(tf.concat(rnn_outputs, axis=2, name='bidirectional_concat_outputs'),
                                             [-1, 2 * max_document_length * self.num_hidden])

        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(rnn_output_flat, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)
            self.fc_output = fc_output

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
