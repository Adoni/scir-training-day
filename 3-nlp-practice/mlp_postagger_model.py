import tensorflow as tf
import numpy
import copy


class mlp_postagger_model:
    def __init__(self, config):
        self.win_size = config['window_size']
        self.embedding_size = config['embedding_size']
        self.vocabulary_size = config['vocabulary_size']
        self.embeddings = tf.Variable(tf.random_uniform(
            [self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        self.layer_size = [self.win_size + 1 + self.win_size]
        for lsize in config['layer_size']:
            self.layer_size.append(lsize)
        self.W = []
        self.b = []
        for i in range(0, len(self.layer_size) - 1):
            Wi = tf.Variable(tf.random_normal([self.layer_size[i],
                                               self.layer_size[i + 1]]))
            self.W.append(Wi)
            bi = tf.Variable(tf.random_normal([self.layer_size[i + 1]]))
            self.b.append(bi)
        self.words = tf.placeholder("int32",
                                    [None, self.win_size + 1 + self.win_size])
        a = []
        z = []
        array_x = tf.nn.embedding_lookup(self.embeddings, self.words)
        x = tf.reshape(array_x, [-1, (self.win_size + 1 + self.win_size) *
                                 self.embedding_size])
        a.append(x)
        for i in range(len(self.layer_size)):
            z.append(tf.add(tf.matmul(a[i], self.W[i]), self.b[i]))
            a.append(tf.nn.relu(z[-1]))
        y = tf.nn.softmax(z[-1])
        self.y_ = tf.placeholder("float32", [None, len(self.layer_size[-1])])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z[-1],
                                                                      self.y_))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=1.5).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.predicted_label = tf.argmax(y, 1)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self, words, taggers):
        self.sess.run(optimizer, feed_dict={self.words: words, y_: taggers})
        loss = self.sess.run(loss, feed_dict={self.words: words, y_: taggers})
        return loss

    def develop(self, dev_data_x, dev_data_y):
        return self.accuracy.eval(feed_dict={self.words: dev_data_x,
                                             self.y_: dev_data_y})

    def preidict(self, test_data_x):
        self.sess.run(self.predicted_label,
                      feed_dict={self.words: test_data_x})
        return label = numpy.argmax(test_data_x, 1)
