import tensorflow as tf
import argparse
import os
import pickle as pkl
import time
import numpy as np
from model.word_rnn import WordRNN
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia
from data_helper import write_csv_files


# NUM_CLASS = 2
# BATCH_SIZE = 64
# NUM_EPOCHS = 30
# MAX_DOCUMENT_LEN = 100
# MAX_DATA_NUM = 8000
# output_csv_dir = 'data/ACL/newsDataset/0bit_1bit_test'
# train_text_dirs = [output_csv_dir + '/0bit_train.txt', output_csv_dir + '/1bit_train.txt']
# train_labels = [0, 1]
# test_text_dirs = [output_csv_dir + '/0bit_test.txt', output_csv_dir + '/1bit_test.txt']
# test_labels = [0, 1]
# train_file = "train.csv"
# test_file = "test.csv"

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
def train(train_x, train_y, test_x, test_y, vocabulary_size, args):
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.num_epochs
        model = WordRNN(vocabulary_size, args.max_document_len, len(args.labels))

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(args.lr)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load variables from pre-trained model
        if not args.pre_trained == "none":
            pre_trained_variables = [v for v in tf.global_variables()
                                     if (v.name.startswith("embedding") or v.name.startswith(
                    "rnn")) and "Adam" not in v.name]
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

        def train_step(batch_x, batch_y):
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y,
                model.keep_prob: 0.5
            }

            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                with open(os.path.join(args.summary_dir, "accuracy.txt"), "a") as f:
                    print("step {0} : loss = {1}".format(step, loss), file=f)
                print("step {0} : loss = {1}".format(step, loss))

        def test_accuracy(test_x, test_y):
            test_batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
            outputs = []
            predictions = []
            for test_batch_x, test_batch_y in test_batches:
                accuracy, prediction = sess.run([model.accuracy, model.predictions],
                                                feed_dict={model.x: test_batch_x, model.y: test_batch_y,
                                                           model.keep_prob: 1.0})
                predictions.extend(prediction.tolist())
                outputs.extend(test_batch_y.tolist())
            labels = np.unique(outputs)
            labels_count_TP = np.array([np.sum(b.astype(int)) for b in
                                        [np.logical_and(np.equal(outputs, label_x), np.equal(predictions, label_x)) for
                                         label_x in labels]])
            labels_count_TN = np.array([np.sum(b.astype(int)) for b in [
                np.logical_not(np.logical_or(np.equal(outputs, label_x), np.equal(predictions, label_x))) for label_x in
                labels]])
            labels_count_FP = np.array([np.sum(b.astype(int)) for b in [
                np.logical_and(np.logical_not(np.equal(outputs, label_x)), np.equal(predictions, label_x)) for label_x
                in
                labels]])
            labels_count_FN = np.array([np.sum(b.astype(int)) for b in [
                np.logical_and(np.equal(outputs, label_x), np.logical_not(np.equal(predictions, label_x))) for label_x
                in
                labels]])
            precisions = labels_count_TP / (labels_count_TP + labels_count_FP)
            recalls = labels_count_TP / (labels_count_TP + labels_count_FN)
            fscores = 2 * precisions * recalls / (precisions + recalls)
            accuracies = (labels_count_TP + labels_count_TN) / (
                labels_count_TP + labels_count_TN + labels_count_FP + labels_count_FN)
            specificities = labels_count_TN / (labels_count_TN + labels_count_FP)
            all_accuracy = np.sum(labels_count_TP) / len(outputs)

            # with open(os.path.join(args.summary_dir, "accuracy.txt"), "a") as f:
            #     print("step %d: test_accuracy=%f"%(step,sum_accuracy / cnt), file=f)

            return precisions, recalls, fscores, accuracies, specificities, all_accuracy, outputs, predictions

        def write_accuracy(precisions, recalls, fscores, accuracies, specificities, all_accuracy, step):
            with open(os.path.join(args.summary_dir, "accuracy.txt"), "a") as f:
                print(
                    "step %d: precision: %s, recall: %s, fscore: %s, accuracy: %s, specificity: %s, all_accuracy: %s" % (
                        step, str(precisions), str(recalls), str(fscores), str(accuracies), str(specificities),
                        str(all_accuracy)), file=f)
            print(
                "step %d: precision: %s, recall: %s, fscore: %s, accuracy: %s, specificity: %s, all_accuracy: %s" % (
                    step, str(precisions), str(recalls), str(fscores), str(accuracies), str(specificities),
                    str(all_accuracy)))
            return

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)

        for batch_x, batch_y in batches:
            train_step(batch_x, batch_y)
            step = tf.train.global_step(sess, global_step)

            if step % 200 == 0:
                test_p, test_r, test_f, test_a, test_s, test_aa, _, _ = test_accuracy(test_x, test_y)
                write_accuracy(test_p, test_r, test_f, test_a, test_s, test_aa, step)
        test_p, test_r, test_f, test_a, test_s, test_aa, labels, predictions = test_accuracy(test_x, test_y)
        write_accuracy(test_p, test_r, test_f, test_a, test_s, test_aa, step)
        with open(os.path.join(args.summary_dir, "LabelsAndPredictions"), "wb") as f:
            final_result = {'labels': labels, 'predictions': predictions}
            pkl.dump(final_result, f)


def logout_config(args, train_y, test_y):
    with open(os.path.join(args.summary_dir, "accuracy.txt"), "w") as f:
        # print("NUM_CLASS=%d, BATCH_SIZE=%d, NUM_EPOCH=%d, MAX_LEN=%d" % (
        #     NUM_CLASS, BATCH_SIZE, NUM_EPOCHS, MAX_DOCUMENT_LEN), file=f)
        # print("dataset_dir: ", output_csv_dir, file=f)
        print(str(args), file=f)

        labels = list(set(train_y))
        labels.sort()
        print("train samples: %d" % len(train_y), file=f)
        for label in labels:
            print("\t class %d in train set: %d samples" % (label, train_y.count(label)), file=f)

        labels = list(set(test_y))
        labels.sort()
        print("test samples: %d" % len(test_y), file=f)
        for label in labels:
            print("\t class %d in test set: %d samples" % (label, test_y.count(label)), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained", type=str, default="auto_encoder", help="none | auto_encoder | language_model")
    parser.add_argument("--data_folder", type=str, default="ACL", help="ACL | Markov | huffman_tree | two_tree")
    parser.add_argument("--data_type", type=str, default="news", help="movie | news | tweet")
    parser.add_argument("--unlabeled_data_num", type=int, default=50000,
                        help="how many unlabeled data samples was used in pretrain")
    parser.add_argument("--labeled_data_num", type=int, default=8000, help="train data samples for each label")
    parser.add_argument("--test_data_num", type=int, default=2000, help="test data samples for each label")
    parser.add_argument("--labels", nargs='+', type=int, default=[0, 1], help="classes to classify")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=40, help="epoch num")
    parser.add_argument("--max_document_len", type=int, default=100, help="max length of sentence")
    args = parser.parse_args()

    dataset_dir = os.path.join("dataset", args.data_folder, args.data_type)
    train_text_dirs = []
    test_text_dirs = []
    if not os.path.exists(os.path.join(dataset_dir, 'train')):
        os.makedirs(os.path.join(dataset_dir, 'train'))
    if not os.path.exists(os.path.join(dataset_dir, 'test')):
        os.makedirs(os.path.join(dataset_dir, 'test'))
    for label in args.labels:
        train_text_dir = os.path.join(dataset_dir, 'train',
                                      args.data_type + '_' + str(label) + 'bit_' + str(args.labeled_data_num) + '.txt')
        test_text_dir = os.path.join(dataset_dir, 'test',
                                     args.data_type + '_' + str(label) + 'bit_' + str(args.test_data_num) + '.txt')
        train_text_dirs.append(train_text_dir)
        test_text_dirs.append(test_text_dir)
        if os.path.exists(train_text_dir) and os.path.exists(test_text_dir):
            continue
        with open(os.path.join(dataset_dir, args.data_type + '_' + str(label) + 'bit.txt'), 'r', encoding='utf8') as f:
            all_lines = f.readlines()
        if not os.path.exists(train_text_dir):
            with open(train_text_dir, 'w', encoding='utf8') as f_train:
                f_train.writelines(all_lines[:args.labeled_data_num])
        if not os.path.exists(test_text_dir):
            with open(test_text_dir, 'w', encoding='utf8') as f_test:
                f_test.writelines(all_lines[-args.test_data_num:])

    model_dir = os.path.join(args.pre_trained, args.data_folder, args.data_type, str(args.unlabeled_data_num))
    path = os.path.join(model_dir, 'bit_'.join([str(x) for x in args.labels]) + 'bit_' + str(args.labeled_data_num))
    if os.path.exists(path) is not True:
        os.makedirs(path)
    args.summary_dir = path
    args.model_dir = model_dir

    write_csv_files(train_text_dirs, test_text_dirs, args.labels, args.labels, path, 'train.csv', 'test.csv',
                    args.labeled_data_num, args.test_data_num)
    train_path = os.path.join(path, 'train.csv')
    test_path = os.path.join(path, 'test.csv')
    print("\nBuilding dictionary..")
    if args.pre_trained == 'none':
        word_dict = build_word_dict(model_dir, None, train_path)
    else:
        word_dict = build_word_dict(model_dir, None)
    print("Preprocessing dataset..")
    label_map = dict()
    k = 0
    for label in args.labels:
        label_map[label] = k
        k = k + 1
    train_x, train_y = build_word_dataset(train_path, test_path, "train", word_dict, args.max_document_len, label_map)
    test_x, test_y = build_word_dataset(train_path, test_path, "test", word_dict, args.max_document_len, label_map)
    logout_config(args, train_y, test_y)
    train(train_x, train_y, test_x, test_y, len(word_dict), args)
