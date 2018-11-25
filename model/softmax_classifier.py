import tensorflow as tf


class SoftmaxClassifier(object):
    def __init__(self, embedding_size, num_class):
        self.embedding_size = embedding_size
        self.fc_num_hidden = 256

        self.x = tf.placeholder(tf.float32, [None, embedding_size])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])

        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(self.x, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)

        with tf.name_scope("output"):
            # self.logits = tf.layers.dense(dropout, num_class, activation=tf.nn.relu)
            self.logits = tf.layers.dense(dropout, num_class)
            # self.logits = tf.layers.dense(self.x, num_class)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
