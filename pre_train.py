import tensorflow as tf
import argparse
import os
from model.auto_encoder import AutoEncoder
from model.language_model import LanguageModel
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia
from data_helper import write_csv_file


# BATCH_SIZE = 64
# NUM_EPOCHS = 10
# MAX_DOCUMENT_LEN = 100
# MAX_UNLABEL_DATA_NUM = 50000
# dataset_dir = 'data/Markov/newsDataset'
# train_text_dirs = [dataset_dir + '/unlabeled.txt', dataset_dir + '/0bit.txt', dataset_dir + '/5bit.txt']
# train_labels = [0, 0, 0]  # unsupervised
# output_csv_dir = dataset_dir + '/0bit_5bit_test'
# train_file = "train_" + str(MAX_UNLABEL_DATA_NUM) + ".csv"

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

def train(train_x, train_y, word_dict, args, model_dir):
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        if args.model == "auto_encoder":
            model = AutoEncoder(word_dict, args.max_document_len)
        elif args.model == "language_model":
            model = LanguageModel(word_dict, args.max_document_len)
        else:
            raise ValueError("Invalid model: {0}. Use auto_encoder | language_model".format(args.model))

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
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

        # Checkpoint
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x):
            feed_dict = {model.x: batch_x}
            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                with open(os.path.join(model_dir, "loss.txt"), "a") as f:
                    print("step {0} : loss = {1}".format(step, loss), file=f)
                print("step {0} : loss = {1}".format(step, loss))

        # Training loop
        batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)

        for batch_x, _ in batches:
            train_step(batch_x)
            step = tf.train.global_step(sess, global_step)

            if step % 5000 == 0:
                if os.path.exists(model_dir) is False:
                    os.makedirs(model_dir)
                saver.save(sess, os.path.join(model_dir, "model.ckpt"), global_step=step)
        if os.path.exists(model_dir) is False:
            os.makedirs(model_dir)
        saver.save(sess, os.path.join(model_dir, "model.ckpt"), global_step=step)


def logout_config(args, model_dir, dict_size):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, "loss.txt"), "w") as f:
        # print("BATCH_SIZE=%d, NUM_EPOCH=%d, MAX_LEN=%d, MAX_UNLABEL_DATA_NUM=%d, MAX_DICT_SIZE=%d" % (
        #     BATCH_SIZE, NUM_EPOCHS, MAX_DOCUMENT_LEN, MAX_UNLABEL_DATA_NUM, args.dict_size), file=f)
        # print("dataset_dir: ", os.path.join(output_csv_dir, args.model + '_' + train_file), file=f)
        # print("train samples: %d" % len(train_y), file=f)
        print(str(args), file=f)
        print("generated dictionary size: %d" % dict_size, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="language_model", help="auto_encoder | language_model")
    # parser.add_argument("--model_name", type=str, default="model", help="the folder name of the model")
    parser.add_argument("--dict_size", type=int, default=20000, help="the max size of word dictionary")
    parser.add_argument("--data_folder", type=str, default="ACL", help="ACL | Markov | huffman_tree | two_tree")
    parser.add_argument("--data_type", type=str, default="news", help="movie | news | tweet")
    parser.add_argument("--unlabeled_data_num", type=int, default=50000, help="how many unlabeled data samples to use")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="epoch num")
    parser.add_argument("--max_document_len", type=int, default=30, help="max length of sentence")
    args = parser.parse_args()

    dataset_dir = os.path.join("dataset", args.data_folder, args.data_type)
    unlabeled_text_dirs = [os.path.join(dataset_dir, args.data_type + '.txt')]
    model_dir = os.path.join(args.model, args.data_folder, args.data_type,
                             str(args.unlabeled_data_num))
    unlabeled_csv_file = 'unlabeled_' + str(args.unlabeled_data_num) + '.csv'
    unlabeled_csv_path = os.path.join(model_dir, unlabeled_csv_file)
    if not os.path.exists(unlabeled_csv_path):
        write_csv_file(unlabeled_text_dirs, [-1], model_dir, unlabeled_csv_file, args.unlabeled_data_num)
    print("\nBuilding dictionary..")
    word_dict = build_word_dict(model_dir, args.dict_size, unlabeled_csv_path)
    print("Preprocessing dataset..")
    train_x, train_y = build_word_dataset(unlabeled_csv_path, None, "train",
                                          word_dict,
                                          args.max_document_len)
    logout_config(args, model_dir, len(word_dict))
    train(train_x, train_y, word_dict, args, model_dir)
