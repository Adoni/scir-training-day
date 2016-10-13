from dataset import read_dataset
import tensorflow as tf
import numpy
import copy


def construct_training_data(dataset, window_size, word_map, tagger_map):
    data_x = []
    data_y = []
    for k in range(len(dataset)):
        train_words = copy.copy(dataset[k][0])
        train_taggers = copy.copy(dataset[k][1])
        for i in range(window_size):
            train_words.insert(0, 'BBBBB')
            train_words.append('BBBBB')
        for i in range(window_size, len(train_words) - window_size):
            train_words_id = []
            for j in range(i - window_size, i + window_size + 1):
                if train_words[j] not in word_map:
                    train_words[j] = 'UNKNOWNWORD'
                train_words_id.append(word_map[train_words[j]])
            assert (len(train_words_id) == window_size + 1 + window_size)
            train_tagger = numpy.zeros(len(tagger_map))
            train_tagger[tagger_map[train_taggers[i - window_size]]] = 1
            data_x.append(train_words_id)
            data_y.append(train_tagger)
    return data_x, data_y


def main():
    train_data = read_dataset('./penn.train.pos.gz')
    print(len(train_data))
    word_map = dict()
    tagger_map = dict()
    for d in train_data:
        for w in d[0]:
            if w not in word_map:
                word_map[w] = len(word_map)
        for t in d[1]:
            if t not in tagger_map:
                tagger_map[t] = len(tagger_map)
    word_map['BBBBB'] = len(word_map)
    word_map['UNKNOWNWORD'] = len(word_map)

    window_size = 2
    train_data_x, train_data_y = construct_training_data(
        train_data, window_size, word_map, tagger_map)
    dev_data_x, dev_data_y = construct_training_data(
        read_dataset('./penn.devel.pos.gz'), window_size, word_map, tagger_map)
    #dev_data_x=train_data_x[0:3000]
    #dev_data_y=train_data_y[0:3000]
    embedding_size = 100

    embeddings = tf.Variable(tf.random_uniform(
        [len(word_map), embedding_size], -1.0, 1.0))
    W = tf.Variable(tf.random_normal([embedding_size * (
        window_size + 1 + window_size), len(tagger_map)]))
    b = tf.Variable(tf.random_normal([len(tagger_map)]))

    words = tf.placeholder("int32", [None, window_size + 1 + window_size])
    array_x = tf.nn.embedding_lookup(embeddings, words)
    x = tf.reshape(array_x, [-1,
                             (window_size + 1 + window_size) * embedding_size])
    h = tf.add(tf.matmul(x, W), b)
    y = tf.nn.softmax(h)
    y_ = tf.placeholder("float32", [None, len(tagger_map)])
    #loss = tf.reduce_mean(tf.square(y - y_))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, y_))
    #loss=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate=1.5).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predicted_label = tf.argmax(y, 1)

    init = tf.initialize_all_variables()
    print('Train Size:', len(train_data_x))
    print('Dev Size:', len(dev_data_x))
    with tf.Session() as sess:
        sess.run(init)
        print("test accuracy %g" % accuracy.eval(feed_dict={
            words: dev_data_x,
            y_: dev_data_y
        }))
        for kkk in range(100):
            avg_loss = 0.0
            for i in range(0, len(train_data_x), batch_size):
                sess.run(optimizer,
                         feed_dict={words: train_data_x[i:i + batch_size],
                                    y_: train_data_y[i:i + batch_size]})
                avg_loss += sess.run(
                    loss,
                    feed_dict={words: train_data_x[i:i + batch_size],
                               y_: train_data_y[i:i + batch_size]})
            #print(kkk, avg_loss)
            print("test accuracy %g" % accuracy.eval(feed_dict={
                words: dev_data_x,
                y_: dev_data_y
            }))
            print(list(sess.run(predicted_label,
                                feed_dict={words: dev_data_x[0:10]})))
            print(numpy.argmax(dev_data_y[0:10], 1))


if __name__ == '__main__':
    main()
