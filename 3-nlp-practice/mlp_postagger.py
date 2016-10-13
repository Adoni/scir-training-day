from mlp_postagger_model import mlp_postagger_model
import copy
import numpy
from dataset import read_dataset


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

    config = dict()
    config['window_size'] = 2
    config['embedding_size'] = 100
    config['layer_size'] = [1 + 2 * config['window_size'], len(tagger_map)]
    config['batch_size'] = 500
    config['dev_every_i'] = 10000
    train_data_x, train_data_y = construct_training_data(
        train_data, config['window_size'], word_map, tagger_map)
    dev_data_x, dev_data_y = construct_training_data(
        read_dataset('./penn.devel.pos.gz'), config['window_size'], word_map,
        tagger_map)
    test_data_x, _ = construct_training_data(
        read_dataset('./penn.devel.pos.gz'), config['window_size'], word_map,
        tagger_map)

    model = mlp_postagger_model(config)

    print('Train Size:', len(train_data_x))
    print('Dev Size:', len(dev_data_x))
    print("Initial dev accuracy %g" % model.develop(dev_data_x, dev_data_y))
    avg_loss = 0.0
    begin = 0
    done = False
    total_trained_instances = 0.0
    last_dev = 0.0
    while (not done):
        end = begin + config['batch_size']
        if end > len(train_data_x):
            end = len(train_data_x)
        avg_loss += model.train(train_data_x[begin:end],
                                train_data_y[begin:end])
        total_trained_instances += end - begin
        if total_trained_instances - last_dev > config['dev_every_i']:
            dev_accuracy = model.develop(dev_data_x, dev_data_y)
            print('Ration %0.2f\tAccuracy %f' %
                  (total_trained_instances / len(train_data_x), dev_accuracy))
            labels = model.preidict(test_data_x)
            last_dev = total_trained_instances
        begin = end
        if begin >= len(train_data_x):
            begin = 0


if __name__ == '__main__':
    main()
