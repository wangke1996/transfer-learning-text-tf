# python example to infer document vectors from trained doc2vec model
import gensim.models as g
import tensorflow as tf
import numpy as np
import argparse
import os
import pickle as pkl
from random import shuffle
from model.softmax_classifier import SoftmaxClassifier
from data_utils import batch_iter


# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

def train(train_x, train_y, test_x, test_y, args):
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.num_epochs
        model = SoftmaxClassifier(len(train_x[0]), len(args.labels))

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(args.lr)
        # optimizer=tf.train.GradientDescentOptimizer(args.lr)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)


        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x, batch_y):
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y,
                model.keep_prob: 0.5
            }

            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 1000 == 0:
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

            if step % 1000 == 0:
                test_p, test_r, test_f, test_a, test_s, test_aa, _, _ = test_accuracy(test_x, test_y)
                write_accuracy(test_p, test_r, test_f, test_a, test_s, test_aa, step)
        test_p, test_r, test_f, test_a, test_s, test_aa, labels, predictions = test_accuracy(test_x, test_y)
        write_accuracy(test_p, test_r, test_f, test_a, test_s, test_aa, step)
        with open(os.path.join(args.summary_dir, "LabelsAndPredictions"), "wb") as f:
            final_result = {'labels': labels, 'predictions': predictions}
            pkl.dump(final_result, f)


def build_dataset(text_dirs, label_map, args, m, type='train'):
    inputs = []
    outputs = []
    pkl_folder = os.path.join(args.model_dir, type)
    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)
    for file_path, label in zip(text_dirs, args.labels):
        file_name = os.path.split(file_path)[-1]
        pkl_path = os.path.join(pkl_folder, file_name + '.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                x = pkl.load(f)
                y = pkl.load(f)
        else:
            with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
                x = f.readlines()
            x = list(map(lambda d: d.strip().split(), x))
            x = list(map(lambda d: m.infer_vector(d, alpha=args.start_lr, steps=args.infer_epochs), x))
            x = list(map(lambda d: d.tolist(), x))
            y = [label] * len(x)
            with open(pkl_path, 'wb') as f:
                pkl.dump(x, f)
                pkl.dump(y, f)
        inputs.extend(x)
        outputs.extend([label_map[l] for l in y])
    samples = list(zip(inputs, outputs))
    shuffle(samples)
    inputs, outputs = zip(*samples)
    return list(inputs), list(outputs)


def logout_config(args, train_y, test_y):
    with open(os.path.join(args.summary_dir, "accuracy.txt"), "w") as f:
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
    parser.add_argument("--method", type=str, default="PVDM", help="PVDM | PVDBOW")
    parser.add_argument("--data_folder", type=str, default="ACL", help="ACL | Markov | huffman_tree | two_tree")
    parser.add_argument("--data_type", type=str, default="news", help="movie | news | tweet")
    parser.add_argument("--unlabeled_data_num", type=int, default=50000,
                        help="how many unlabeled data samples was used in pretrain")
    parser.add_argument("--labeled_data_num", type=int, default=8000, help="train data samples for each label")
    parser.add_argument("--test_data_num", type=int, default=2000, help="test data samples for each label")
    parser.add_argument("--labels", nargs='+', type=int, default=[0, 1], help="classes to classify")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training classifier")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=150, help="epoch num for training classifier")
    parser.add_argument("--infer_epochs", type=int, default=1000, help="epoch num for inferring sentence vector")
    parser.add_argument("--start_lr", type=float, default=0.01, help="start learning rate for inferring")
    args = parser.parse_args()
    args.pre_trained = "doc2vec"

    dataset_dir = os.path.join("dataset", args.data_folder, args.data_type)
    train_text_dirs = []
    test_text_dirs = []
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

    model_dir = os.path.join(args.pre_trained, args.data_folder, args.data_type, str(args.unlabeled_data_num),
                             args.method)
    path = os.path.join(model_dir, 'bit_'.join([str(x) for x in args.labels]) + 'bit_' + str(args.labeled_data_num))
    if os.path.exists(path) is not True:
        os.makedirs(path)
    args.summary_dir = path
    args.model_dir = model_dir

    label_map = dict()
    k = 0
    for label in args.labels:
        label_map[label] = k
        k = k + 1
    print("loading model...")
    m = g.Doc2Vec.load(os.path.join(model_dir, 'model.bin'))
    print("inferring doc vectors and preparing dataset...")
    train_x, train_y = build_dataset(train_text_dirs, label_map, args, m, 'train')
    test_x, test_y = build_dataset(test_text_dirs, label_map, args, m, 'test')
    logout_config(args, train_y, test_y)
    print("begin training...")
    train(train_x, train_y, test_x, test_y, args)


