import tensorflow as tf
import argparse
import os
from model.auto_encoder import AutoEncoder
from model.language_model import LanguageModel
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia
from data_helper import write_csv_file

BATCH_SIZE = 64
NUM_EPOCHS = 10
MAX_DOCUMENT_LEN = 100
MAX_UNLABEL_DATA_NUM = 50000
dataset_dir = 'data/movieDataset'
train_text_dirs = [dataset_dir + '/unlabeled.txt', dataset_dir + '/0bit.txt', dataset_dir + '/1bit.txt']
train_labels = [0, 0, 0]  # unsupervised
output_csv_dir = dataset_dir + '/0bit_1bit_test'
train_file = "language_model_train_"+str(MAX_UNLABEL_DATA_NUM)+".csv"


def train(train_x, train_y, word_dict, args):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if args.model == "auto_encoder":
            model = AutoEncoder(word_dict, MAX_DOCUMENT_LEN)
        elif args.model == "language_model":
            model = LanguageModel(word_dict, MAX_DOCUMENT_LEN)
        else:
            raise ValueError("Invalid model: {0}. Use auto_encoder | language_model".format(args.model))

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
        summary_writer = tf.summary.FileWriter(os.path.join(args.model, args.model_name), sess.graph)

        # Checkpoint
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x):
            feed_dict = {model.x: batch_x}
            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                with open(os.path.join(args.model, args.model_name, "loss.txt"), "a") as f:
                    print("step {0} : loss = {1}".format(step, loss), file=f)
                print("step {0} : loss = {1}".format(step, loss))

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)

        for batch_x, _ in batches:
            train_step(batch_x)
            step = tf.train.global_step(sess, global_step)

            if step % 5000 == 0:
                dir = os.path.join(args.model, args.model_name)
                if os.path.exists(dir) is False:
                    os.makedirs(dir)
                saver.save(sess, os.path.join(dir, "model.ckpt"), global_step=step)


def logout_config(args, train_y, dict_size):
    dir = os.path.join(args.model, args.model_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, "loss.txt"), "w") as f:
        print("BATCH_SIZE=%d, NUM_EPOCH=%d, MAX_LEN=%d, MAX_UNLABEL_DATA_NUM=%d, MAX_DICT_SIZE=%d" % (
            BATCH_SIZE, NUM_EPOCHS, MAX_DOCUMENT_LEN, MAX_UNLABEL_DATA_NUM, args.dict_size), file=f)
        print("dataset_dir: ", os.path.join(output_csv_dir, train_file), file=f)
        print("train samples: %d" % len(train_y), file=f)
        print("generated dictionary size: %d" % dict_size, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="auto_encoder", help="auto_encoder | language_model")
    parser.add_argument("--model_name", type=str, default="model", help="the folder name of the model")
    parser.add_argument("--dict_size", type=int, default=20000, help="the max size of word dictionary")
    args = parser.parse_args()

    if not os.path.exists("dbpedia_csv"):
        print("Downloading dbpedia dataset...")
        download_dbpedia()

    write_csv_file(train_text_dirs, train_labels, output_csv_dir, train_file, MAX_UNLABEL_DATA_NUM)
    print("\nBuilding dictionary..")
    dict_src_path = os.path.join(output_csv_dir, train_file)
    word_dict = build_word_dict(os.path.join(args.model, args.model_name), args.dict_size, dict_src_path)
    print("Preprocessing dataset..")
    train_x, train_y = build_word_dataset(os.path.join(output_csv_dir, train_file), None, "train", word_dict,
                                          MAX_DOCUMENT_LEN)
    logout_config(args, train_y, len(word_dict))
    train(train_x, train_y, word_dict, args)
