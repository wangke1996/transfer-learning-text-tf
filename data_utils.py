import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np

TRAIN_DICT_PATH = "dbpedia_csv/train.csv"


def download_dbpedia():
    dbpedia_url = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'

    wget.download(dbpedia_url)
    with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
        tar.extractall()


def clean_str(text):
    if re.match('[\u4e00-\u9fa5]+', text) is not None:
        if len(text)%4==3:
            text=text+'。'
        text = ' '.join(text)
    else:
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    return text


def build_word_dict(dict_dir, vocabulary_size=None, dict_src_path=TRAIN_DICT_PATH):
    if not os.path.exists(dict_dir):
        os.makedirs(dict_dir)
    dict_path = os.path.join(dict_dir, "word_dict.pickle")
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            word_dict = pickle.load(f)
        if vocabulary_size is None or len(word_dict) == vocabulary_size:
            print("use word dictionary at %s, vocabulary size: %d" % (dict_path, len(word_dict)))
            return word_dict
    train_df = pd.read_csv(dict_src_path, names=["class", "title", "content"])
    contents = train_df["content"]

    words = list()
    for content in contents:
        for word in word_tokenize(clean_str(content)):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<s>"] = 2
    word_dict["</s>"] = 3
    for word, count in word_counter:
        if vocabulary_size is None or len(word_dict) != vocabulary_size:
            word_dict[word] = len(word_dict)

    with open(dict_path, "wb") as f:
        pickle.dump(word_dict, f)
    print("build dictionary from %s, vocabulary size: %d" % (dict_path, len(word_dict)))

    return word_dict


def build_word_dataset(train_path, test_path, step, word_dict, document_max_len, label_map=None):
    if step == "train":
        df = pd.read_csv(train_path, names=["class", "title", "content"])
    else:
        df = pd.read_csv(test_path, names=["class", "title", "content"])

    # Shuffle dataframe
    df = df.sample(frac=1)
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["content"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

    # y = list(map(lambda d: d - 1, list(df["class"])))
    y = list(map(lambda d: d, list(df["class"])))
    if label_map is not None:
        y = [label_map[i] for i in y]
    return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
