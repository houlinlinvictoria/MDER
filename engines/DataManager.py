# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: DataManager.py
# @Project: sequence-lableing-vex

import os, logging
import numpy as np
from engines.utils import read_csv_
import jieba
jieba.setLogLevel(logging.INFO)


class DataManager:
    def __init__(self, configs, logger):
        self.configs=configs
        self.train_file = configs.train_file
        self.logger = logger
        self.hyphen = configs.hyphen

        self.UNKNOWN = "<UNK>"
        self.PADDING = "<PAD>"

        self.train_file = configs.datasets_fold + "/" + configs.train_file
        self.dev_file = configs.datasets_fold + "/" + configs.dev_file
        self.test_file = configs.datasets_fold + "/" + configs.test_file
        self.trainr_file = configs.datasets_fold + "/" + configs.trainr_file
        self.devr_file = configs.datasets_fold + "/" + configs.devr_file
        self.testr_file = configs.datasets_fold + "/" + configs.testr_file


        self.output_test_file = configs.datasets_fold + "/" + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + "/" + configs.output_sentence_entity_file

        self.label_scheme = configs.label_scheme
        self.label_level = configs.label_level
        self.suffix = configs.suffix
        self.labeling_level = configs.labeling_level

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim

        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = self.vocabs_dir + "/token2id"
        self.label2id_file = self.vocabs_dir + "/label2id"
        self.rule2id_file = self.vocabs_dir + "/rule2id"

        self.token2id, self.id2token, self.label2id, self.id2label, self.rule2id, self.id2rule = self.loadVocab()

        self.max_token_number = len(self.token2id)+1
        self.max_label_number = len(self.label2id)
        self.max_rule_number = len(self.rule2id)

        jieba.load_userdict(self.token2id.keys())

        self.logger.info("dataManager initialed...\n")

    def loadVocab(self):
        if not os.path.isfile(self.token2id_file):
            self.logger.info("vocab files not exist, building vocab...")
            return self.buildVocab(self.train_file, self.trainr_file)

        self.logger.info("loading vocab...")
        token2id = {}
        id2token = {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id = {}
        id2label = {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                label = row.split('\t')[0]
                label_id = int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label

        rule2id = {}
        id2rule = {}
        with open(self.rule2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                rule = row.split('\t')[0]
                rule_id = int(row.split('\t')[1])
                rule2id[rule] = rule_id
                id2rule[rule_id] = rule

        return token2id, id2token, label2id, id2label, rule2id, id2rule

    def buildVocab(self, train_path, trainr_path):
        df_train = read_csv_(train_path, names=["token", "label"],delimiter=self.configs.delimiter)
        df_trainr = read_csv_(trainr_path, names=["token", "rule"],delimiter=self.configs.delimiter)
        tokens = list(set(df_train["token"][df_train["token"].notnull()]))
        labels = list(set(df_train["label"][df_train["label"].notnull()]))
        rules = list(set(df_trainr["rule"][df_trainr["rule"].notnull()]))
        rule2id = dict(zip(rules, range(1, len(rules) + 1)))
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2rule = dict(zip(range(1, len(rules) + 1), rules))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        id2rule[0] = self.PADDING
        id2token[0] = self.PADDING
        id2label[0] = self.PADDING
        rule2id[self.PADDING] = 0
        token2id[self.PADDING] = 0
        label2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1

        self.saveVocab(id2token, id2label, id2rule)

        return token2id, id2token, label2id, id2label, rule2id, id2rule

    def saveVocab(self, id2token, id2label, id2rule):
        with open(self.token2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + "\t" + str(idx) + "\n")
        with open(self.label2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + "\t" + str(idx) + "\n")
        with open(self.rule2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2rule:
                outfile.write(id2rule[idx] + "\t" + str(idx) + "\n")

    def getEmbedding(self, embed_file):
        emb_matrix = np.random.normal(loc=0.0, scale=0.08, size=(len(self.token2id.keys()), self.embedding_dim))
        emb_matrix[self.token2id[self.PADDING], :] = np.zeros(shape=(self.embedding_dim))

        with open(embed_file, "r", encoding="utf-8") as infile:
            for row in infile:
                row = row.rstrip()
                items = row.split()
                token = items[0]
                assert self.embedding_dim == len(
                    items[1:]), "embedding dim must be consistent with the one in `token_emb_dir'."
                emb_vec = np.array([float(val) for val in items[1:]])
                if token in self.token2id.keys():
                    emb_matrix[self.token2id[token], :] = emb_vec

        return emb_matrix

    def nextBatch(self, X, y, start_index):
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def nextRandomBatch(self, X, y):
        X_batch = []
        y_batch = []
        for i in range(self.batch_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def padding(self, sample):
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare(self, tokens, labels, is_padding=True, return_psyduo_label=False):
        X = []
        y = []
        y_psyduo = []
        tmp_x = []
        tmp_y = []
        tmp_y_psyduo = []

        for record in zip(tokens, labels):
            c = record[0]
            l = record[1]
            if c == -1:  # empty line
                if len(tmp_x) <= self.max_sequence_length:
                    X.append(tmp_x)
                    y.append(tmp_y)
                    if return_psyduo_label: y_psyduo.append(tmp_y_psyduo)
                tmp_x = []
                tmp_y = []
                if return_psyduo_label: tmp_y_psyduo = []
            else:
                tmp_x.append(c)
                tmp_y.append(l)
                if return_psyduo_label: tmp_y_psyduo.append(self.label2id["O"])
        if is_padding:
            X = np.array(self.padding(X))
        else:
            X = np.array(X)
        y = np.array(self.padding(y))
        if return_psyduo_label:
            y_psyduo = np.array(self.padding(y_psyduo))
            return X, y_psyduo

        return X, y

    def getTrainingSet(self, train_val_ratio=0.9):
        df_train = read_csv_(self.train_file, names=["token", "label"],delimiter=self.configs.delimiter)
        df_trainr = read_csv_(self.trainr_file, names=["token", "label"], delimiter=self.configs.delimiter)


        # map the token and label into id
        df_train["token_id"] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        df_trainr["label_id"] = df_trainr.label.map(lambda x: -1 if str(x) == str(np.nan) else self.rule2id[x])

        # convert the data in maxtrix
        X, y = self.prepare(df_train["token_id"], df_train["label_id"])
        _, r = self.prepare(df_train["token_id"], df_trainr["label_id"])

        # shuffle the samples
        num_samples = len(X)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]
        r = r[indexs]

        if self.dev_file != None:
            X_train = X
            y_train = y
            r_train = r
            X_val, y_val, r_val = self.getValidingSet()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            r_train = r[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]
            r_val = r[int(num_samples * train_val_ratio):]

        self.logger.info("\ntraining set size: %d, validating set size: %d\n" % (len(X_train), len(y_val)))

        return X_train, y_train,r_train, X_val, y_val, r_val

    def getValidingSet(self):
        df_val = read_csv_(self.dev_file, names=["token", "label"],delimiter=self.configs.delimiter)
        df_valr = read_csv_(self.devr_file, names=["token", "label"],delimiter=self.configs.delimiter)

        df_val["token_id"] = df_val.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_val["label_id"] = df_val.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        df_valr["label_id"] = df_valr.label.map(lambda x: -1 if str(x) == str(np.nan) else self.rule2id[x])

        X_val, y_val = self.prepare(df_val["token_id"], df_val["label_id"])
        _, r_val = self.prepare(df_val["token_id"], df_valr["label_id"])
        return X_val, y_val, r_val

    def getTestingSet(self):
        df_test = read_csv_(self.test_file, names=None,delimiter=self.configs.delimiter)
        df_testr = read_csv_(self.testr_file, names=None,delimiter=self.configs.delimiter)

        if len(list(df_test.columns)) == 2:
            df_test.columns = ["token", "label"]
            df_testr.columns = ["token", "label"]
            df_test = df_test[["token"]]
            df_testr = df_testr[["label"]]
        elif len(list(df_test.columns)) == 1:
            df_test.columns = ["token"]
            df_testr.columns = ["token", "label"]
            df_testr = df_testr[["label"]]

        df_test["token_id"] = df_test.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_testr["label_id"] = df_testr.label.map(lambda x: -1 if str(x) == str(np.nan) else self.rule2id[x])
        df_test["token"] = df_test.token.map(lambda x: -1 if str(x) == str(np.nan) else x)

        X_test_id, y_test_psyduo_label = self.prepare(df_test["token_id"], df_test["token_id"],
                                                      return_psyduo_label=True)
        X_test_token, _ = self.prepare(df_test["token"], df_test["token"])
        _, r_test = self.prepare(df_test["token_id"], df_testr["label_id"])

        self.logger.info("\ntesting set size: %d\n" % (len(X_test_id)))
        return X_test_id, y_test_psyduo_label, X_test_token, r_test

    def getTestingrealY_str(self):
        df_test = read_csv_(self.test_file, names=None, delimiter=self.configs.delimiter)

        if len(list(df_test.columns)) == 2:
            df_test.columns = ["token", "label"]
            df_test = df_test[["label"]]
        elif len(list(df_test.columns)) == 1:
            df_test.columns = ["label"]

        df_test["label_id"] = df_test.label.map(lambda x: self.mapFuncLable(x, self.label2id))
        df_test["label"] = df_test.label.map(lambda x: -1 if str(x) == str(np.nan) else x)

        Y_test_id, y_test_psyduo_label = self.prepare(df_test["label_id"], df_test["label_id"],
                                                      return_psyduo_label=True)
        Y_test_label, _ = self.prepare(df_test["label"], df_test["label"])

        # self.logger.info("\ntesting set size: %d\n" % (len(Y_test_id)))
        return Y_test_id, y_test_psyduo_label, Y_test_label

    def mapFunc(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def mapFuncLable(self, x, label2id):
        if str(x) == str(np.nan):
            return -1
        # elif x not in token2id:
        #     return token2id[self.UNKNOWN]
        else:
            return label2id[x]

    def prepare_single_sentence(self, sentence):
        if self.labeling_level == 'word':
            if self.check_contain_chinese(sentence):
                sentence = list(jieba.cut(sentence))
            else:
                sentence = list(sentence.split())
        elif self.labeling_level == 'char':
            sentence = list(sentence)

        gap = self.batch_size - 1

        x_ = []
        y_ = []

        for token in sentence:
            try:
                x_.append(self.token2id[token])
            except:
                x_.append(self.token2id[self.UNKNOWN])
            y_.append(self.label2id["O"])

        if len(x_) < self.max_sequence_length:
            sentence += ['x' for _ in range(self.max_sequence_length - len(sentence))]
            x_ += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x_))]
            y_ += [self.label2id["O"] for _ in range(self.max_sequence_length - len(y_))]
        elif len(x_) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
            x_ = x_[:self.max_sequence_length]
            y_ = y_[:self.max_sequence_length]

        X = [x_]
        Sentence = [sentence]
        Y = [y_]
        X += [[0 for j in range(self.max_sequence_length)] for i in range(gap)]
        Sentence += [['x' for j in range(self.max_sequence_length)] for i in range(gap)]
        Y += [[self.label2id['O'] for j in range(self.max_sequence_length)] for i in range(gap)]
        X = np.array(X)
        Sentence = np.array(Sentence)
        Y = np.array(Y)

        return X, Sentence, Y

    def check_contain_chinese(self, check_str):
        for ch in list(check_str):
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
