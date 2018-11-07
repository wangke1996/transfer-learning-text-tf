# python example to train doc2vec model (with or without pre-trained word embeddings)
import argparse
import os
import gensim.models as g
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="PVDM", help="PVDM | PVDBOW")
    parser.add_argument("--dict_size", type=int, default=20000, help="the max size of word dictionary")
    parser.add_argument("--data_folder", type=str, default="ACL", help="ACL | Markov | huffman_tree | two_tree")
    parser.add_argument("--data_type", type=str, default="news", help="movie | news | tweet")
    parser.add_argument("--unlabeled_data_num", type=int, default=50000, help="how many unlabeled data samples to use")
    parser.add_argument("--embedding_size", type=int, default=256, help="word and doc embedding size")
    parser.add_argument("--num_epochs", type=int, default=300, help="epoch num")
    parser.add_argument("--window_size", type=int, default=15, help="sliding window size")
    parser.add_argument("--concat", type=int, default=1, help="1 for concat word vectors, 0 for sum or average")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Ignores all words with total frequency lower than this")
    parser.add_argument("--sampling_threshold", type=float, default=1e-5,
                        help="the threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5")
    parser.add_argument("--worker_count", type=int, default=1, help="Use these many worker threads to train the model")
    parser.add_argument("--negative_samples", type=int, default=5,
                        help="how many noise words should be drawn in negative sampling")

    args = parser.parse_args()
    args.model = "doc2vec"

    dataset_dir = os.path.join("dataset", args.data_folder, args.data_type)
    unlabeled_text_dir = os.path.join(dataset_dir, args.data_type + '.txt')
    model_dir = os.path.join(args.model, args.data_folder, args.data_type,
                             str(args.unlabeled_data_num),args.method)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.NOTSET,
                        filename=os.path.join(model_dir, "log.txt"))

    train_txt = os.path.join(model_dir, 'unlabeled_' + str(args.unlabeled_data_num) + '.txt')
    if not os.path.exists(train_txt):
        with open(unlabeled_text_dir, 'r', encoding='utf8', errors='ignore') as f:
            all_lines = f.readlines()
        with open(train_txt, 'w', encoding='utf8') as f:
            f.writelines(all_lines[:args.unlabeled_data_num])
    with open(os.path.join(model_dir, 'config_info.txt'), 'w', encoding='utf8') as f:
        print(str(args), file=f)

    docs = g.doc2vec.TaggedLineDocument(train_txt)

    if args.method=="PVDM":
        dm=1
    else:
        dm=0
    model = g.Doc2Vec(docs, vector_size=args.embedding_size, max_vocab_size=args.dict_size, window=args.window_size,
                      min_count=args.min_count, sample=args.sampling_threshold, workers=args.worker_count, hs=0,
                      dm=dm, negative=args.negative_samples, dbow_words=1, dm_concat=args.concat,
                      epochs=args.num_epochs)
    saved_path = os.path.join(model_dir, 'model.bin')
    # save model
    model.save(saved_path)
