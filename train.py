import tensorflow as tf
import argparse
import os
import time
from model.word_rnn import WordRNN
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia
from data_helper import write_csv_files

NUM_CLASS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 30
MAX_DOCUMENT_LEN = 100
MAX_DATA_NUM = 8000
output_csv_dir = 'data/tweetDataset/0bit_1bit_test'
train_text_dirs = [output_csv_dir + '/0bit_train.txt', output_csv_dir + '/1bit_train.txt']
train_labels = [0, 1]
test_text_dirs = [output_csv_dir + '/0bit_test.txt', output_csv_dir + '/1bit_test.txt']
test_labels = [0, 1]
train_file = "train.csv"
test_file = "test.csv"


def train(train_x, train_y, test_x, test_y, vocabulary_size, args):
    with tf.Session() as sess:
        model = WordRNN(vocabulary_size, MAX_DOCUMENT_LEN, NUM_CLASS)

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
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
                    "birnn")) and "Adam" not in v.name]
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state(os.path.join(args.pre_trained, args.model_name))
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

        def test_accuracy(test_x, test_y, step):
            test_batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            for test_batch_x, test_batch_y in test_batches:
                accuracy = sess.run(model.accuracy,
                                    feed_dict={model.x: test_batch_x, model.y: test_batch_y, model.keep_prob: 1.0})
                sum_accuracy += accuracy
                cnt += 1

            # with open(os.path.join(args.summary_dir, "accuracy.txt"), "a") as f:
            #     print("step %d: test_accuracy=%f"%(step,sum_accuracy / cnt), file=f)

            return sum_accuracy / cnt

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)

        for batch_x, batch_y in batches:
            train_step(batch_x, batch_y)
            step = tf.train.global_step(sess, global_step)

            if step % 200 == 0:
                test_acc = test_accuracy(test_x, test_y, step)
                with open(os.path.join(args.summary_dir, "accuracy.txt"), "a") as f:
                    print("test_accuracy = {0}\n".format(test_acc), file=f)
                print("test_accuracy = {0}\n".format(test_acc))


def logout_config(summary_dir, train_y, test_y):
    with open(os.path.join(summary_dir, "accuracy.txt"), "w") as f:
        print("NUM_CLASS=%d, BATCH_SIZE=%d, NUM_EPOCH=%d, MAX_LEN=%d" % (
            NUM_CLASS, BATCH_SIZE, NUM_EPOCHS, MAX_DOCUMENT_LEN), file=f)
        print("dataset_dir: ", output_csv_dir, file=f)

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
    parser.add_argument("--pre_trained", type=str, default="none", help="none | auto_encoder | language_model")
    parser.add_argument("--model_name", type=str, default="model", help="the folder name of the model")
    args = parser.parse_args()
    path = os.path.join('summary', args.model_name + '_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
    if os.path.exists(path) is not True:
        os.makedirs(path)
    args.summary_dir = path

    if not os.path.exists("dbpedia_csv"):
        print("Downloading dbpedia dataset...")
        download_dbpedia()

    write_csv_files(train_text_dirs, test_text_dirs, train_labels, test_labels, output_csv_dir, train_file, test_file,
                    MAX_DATA_NUM)
    print("\nBuilding dictionary..")
    word_dict = build_word_dict(os.path.join(args.pre_trained, args.model_name))
    print("Preprocessing dataset..")
    train_path = os.path.join(output_csv_dir, train_file)
    test_path = os.path.join(output_csv_dir, test_file)
    train_x, train_y = build_word_dataset(train_path, test_path, "train", word_dict, MAX_DOCUMENT_LEN)
    test_x, test_y = build_word_dataset(train_path, test_path, "test", word_dict, MAX_DOCUMENT_LEN)
    logout_config(args.summary_dir, train_y, test_y)
    train(train_x, train_y, test_x, test_y, len(word_dict), args)
