# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Linlin Hou
# @FileName: BiLSTM_CRFs.py
# @Project: sequence-lableing-vex

import math, os
from engines.utils import metrics, save_csv_, extractEntity
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from itertools import chain

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BiLSTM_CRFs(object):
    def __init__(self, configs, logger, dataManager):
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.CUDA_VISIBLE_DEVICES

        self.configs = configs
        self.logger = logger
        self.logdir = configs.log_dir
        self.measuring_metrics = configs.measuring_metrics
        self.dataManager = dataManager

        if configs.mode == "train":
            self.is_training = True
        else:
            self.is_training = False

        self.checkpoint_name = configs.checkpoint_name
        self.checkpoints_dir = configs.checkpoints_dir
        self.output_test_file = configs.datasets_fold + "/" + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + "/" + configs.output_sentence_entity_file
        self.is_real_output_sentence_entity = configs.is_real_output_sentence_entity
        self.real_output_sentence_entity_file = configs.datasets_fold + "/" + configs.real_output_sentence_entity_file
        self.is_compare_output_sentence_entity = configs.is_compare_output_sentence_entity
        self.compare_output_sentence_entity_file = configs.datasets_fold + "/" + configs.compare_output_sentence_entity_file
        self.is_real_test = configs.is_real_test

        self.biderectional = configs.biderectional
        self.cell_type = configs.cell_type
        self.num_layers = configs.encoder_layers

        self.is_crf = configs.use_crf

        self.learning_rate = configs.learning_rate
        self.dropout_rate = configs.dropout
        self.batch_size = configs.batch_size

        self.emb_dim = configs.embedding_dim
        self.hidden_dim = configs.hidden_dim

        if configs.cell_type == 'LSTM':
            if self.biderectional:
                self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            else:
                self.cell = tf.nn.rnn_cell.LSTMCell(2 * self.hidden_dim)
        else:
            if self.biderectional:
                self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            else:
                self.cell = tf.nn.rnn_cell.GRUCell(2 * self.hidden_dim)

        self.is_attention = configs.use_self_attention
        self.attention_dim = configs.attention_dim

        self.num_epochs = configs.epoch
        self.max_time_steps = configs.max_sequence_length

        self.num_tokens = dataManager.max_token_number
        self.num_classes = dataManager.max_label_number

        self.is_early_stop = configs.is_early_stop
        self.patient = configs.patient

        self.max_to_keep = configs.checkpoints_max_to_keep
        self.print_per_batch = configs.print_per_batch
        self.best_f1_val = 0

        if configs.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif configs.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif configs.optimizer == 'RMSprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif configs.optimizer == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.initializer = tf.contrib.layers.xavier_initializer()
        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)

        if configs.use_pretrained_embedding:
            embedding_matrix = dataManager.getEmbedding(configs.token_emb_dir)
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_tokens, self.emb_dim], trainable=True,
                                             initializer=self.initializer)

        self.build()
        self.logger.info("model initialed...\n")

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def build(self):
        self.inputs = tf.placeholder(tf.int32, [None, self.max_time_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.max_time_steps])

        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.max_time_steps, 0)

        # lstm cell
        if self.biderectional:
            lstm_cell_fw = self.cell
            lstm_cell_bw = self.cell

            # dropout
            if self.is_training:
                lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
                lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

            # get the length of each sample
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)

            # forward and backward
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )

        else:
            lstm_cell = self.cell
            if self.is_training:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)

            outputs, _ = tf.contrib.rnn.static_rnn(
                lstm_cell,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )
        # outputs: list_steps[batch, 2*dim]
        outputs = tf.concat(outputs, 1)
        outputs = tf.reshape(outputs, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])

        # self attention module
        if self.is_attention:
            H1 = tf.reshape(outputs, [-1, self.hidden_dim * 2])
            W_a1 = tf.get_variable("W_a1", shape=[self.hidden_dim * 2, self.attention_dim],
                                   initializer=self.initializer, trainable=True)
            u1 = tf.matmul(H1, W_a1)

            H2 = tf.reshape(tf.identity(outputs), [-1, self.hidden_dim * 2])
            W_a2 = tf.get_variable("W_a2", shape=[self.hidden_dim * 2, self.attention_dim],
                                   initializer=self.initializer, trainable=True)
            u2 = tf.matmul(H2, W_a2)

            u1 = tf.reshape(u1, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
            u2 = tf.reshape(u2, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
            u = tf.matmul(u1, u2, transpose_b=True)

            # Array of weights for each time step
            A = tf.nn.softmax(u, name="attention")
            outputs = tf.matmul(A, tf.reshape(tf.identity(outputs),
                                              [self.batch_size, self.max_time_steps, self.hidden_dim * 2]))

        # linear
        self.outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes],
                                         initializer=self.initializer)
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes], initializer=self.initializer)
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        self.logits = tf.reshape(self.logits, [self.batch_size, self.max_time_steps, self.num_classes])
        # print(self.logits.get_shape().as_list())
        if not self.is_crf:
            # softmax
            softmax_out = tf.nn.softmax(self.logits, axis=-1)

            self.batch_pred_sequence = tf.cast(tf.argmax(softmax_out, -1), tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            mask = tf.sequence_mask(self.length)

            self.losses = tf.boolean_mask(losses, mask)

            self.loss = tf.reduce_mean(losses)
        else:
            # crf
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.targets, self.length)
            self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(self.logits,
                                                                                           self.transition_params,
                                                                                           self.length)

            self.loss = tf.reduce_mean(-self.log_likelihood)

        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.dev_summary = tf.summary.scalar("loss", self.loss)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def train(self):
        X_train, y_train, X_val, y_val = self.dataManager.getTrainingSet()
        tf.initialize_all_variables().run(session=self.sess)

        saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.logdir + "/training_loss", self.sess.graph)
        dev_writer = tf.summary.FileWriter(self.logdir + "/validating_loss", self.sess.graph)

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        num_val_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))

        cnt = 0
        cnt_dev = 0
        unprogressed = 0
        very_start_time = time.time()
        best_at_epoch = 0
        self.logger.info("\ntraining starting" + ("+" * 20))
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # shuffle train at each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]

            self.logger.info("\ncurrent epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                X_train_batch, y_train_batch = self.dataManager.nextBatch(X_train, y_train,
                                                                          start_index=iteration * self.batch_size)
                _, loss_train, train_batch_viterbi_sequence, train_summary = \
                    self.sess.run([
                        self.opt_op,
                        self.loss,
                        self.batch_pred_sequence,
                        self.train_summary
                    ],
                        feed_dict={
                            self.inputs: X_train_batch,
                            self.targets: y_train_batch,
                        })

                if iteration % self.print_per_batch == 0:
                    cnt += 1
                    train_writer.add_summary(train_summary, cnt)

                    measures = metrics(X_train_batch, y_train_batch,
                                       train_batch_viterbi_sequence,
                                       self.measuring_metrics, self.dataManager)

                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ": %.3f " % v)
                    self.logger.info("training batch: %5d, loss: %.5f, %s" % (iteration, loss_train, res_str))

            # validation
            loss_vals = list()
            val_results = dict()
            for measu in self.measuring_metrics:
                val_results[measu] = 0

            for iterr in range(num_val_iterations):
                cnt_dev += 1
                X_val_batch, y_val_batch = self.dataManager.nextBatch(X_val, y_val, start_index=iterr * self.batch_size)

                loss_val, val_batch_viterbi_sequence, dev_summary = \
                    self.sess.run([
                        self.loss,
                        self.batch_pred_sequence,
                        self.dev_summary
                    ],
                        feed_dict={
                            self.inputs: X_val_batch,
                            self.targets: y_val_batch,
                        })

                measures = metrics(X_val_batch, y_val_batch, val_batch_viterbi_sequence,
                                   self.measuring_metrics, self.dataManager)
                dev_writer.add_summary(dev_summary, cnt_dev)

                for k, v in measures.items():
                    val_results[k] += v
                loss_vals.append(loss_val)

            time_span = (time.time() - start_time) / 60
            val_res_str = ''
            dev_f1_avg = 0
            for k, v in val_results.items():
                val_results[k] /= num_val_iterations
                val_res_str += (k + ": %.3f " % val_results[k])
                if k == 'f1': dev_f1_avg = val_results[k]

            self.logger.info("time consumption:%.2f(min),  validation loss: %.5f, %s" %
                             (time_span, np.array(loss_vals).mean(), val_res_str))
            if np.array(dev_f1_avg).mean() > self.best_f1_val:
                unprogressed = 0
                self.best_f1_val = np.array(dev_f1_avg).mean()
                best_at_epoch = epoch
                saver.save(self.sess, self.checkpoints_dir + "/" + self.checkpoint_name, global_step=self.global_step)
                self.logger.info("saved the new best model with f1: %.3f" % (self.best_f1_val))
            else:
                unprogressed += 1

            if self.is_early_stop:
                if unprogressed >= self.patient:
                    self.logger.info("early stopped, no progress obtained within %d epochs" % self.patient)
                    self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
                    self.logger.info(
                        "total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
                    self.sess.close()
                    return
        self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
        self.logger.info("total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
        self.sess.close()

    def test(self):
        
        X_test, y_test_psyduo_label, X_test_str = self.dataManager.getTestingSet()
        _, _, Y_test_str = self.dataManager.getTestingrealY_str()

        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        self.logger.info("total number of testing iterations: " + str(num_iterations))

        self.logger.info("loading model parameter\n")
        tf.initialize_all_variables().run(session=self.sess)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoints_dir))

        tokens = []
        labels = []
        real_labels = []
        real_entities = []
        real_entities_types = []
        entities = []
        entities_types = []
        self.logger.info("\ntesting starting" + ("+" * 20))
        for i in range(num_iterations):
            self.logger.info("batch: " + str(i + 1))
            X_test_batch = X_test[i * self.batch_size: (i + 1) * self.batch_size]
            X_test_str_batch = X_test_str[i * self.batch_size: (i + 1) * self.batch_size]
            Y_test_str_batch = Y_test_str[i * self.batch_size: (i + 1) * self.batch_size]
            y_test_psyduo_label_batch = y_test_psyduo_label[i * self.batch_size: (i + 1) * self.batch_size]

            if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                X_test_batch = list(X_test_batch)
                X_test_str_batch = list(X_test_str_batch)
                Y_test_str_batch = list(Y_test_str_batch)
                y_test_psyduo_label_batch = list(y_test_psyduo_label_batch)
                gap = self.batch_size - len(X_test_batch)

                X_test_batch += [[0 for j in range(self.max_time_steps)] for i in range(gap)]
                X_test_str_batch += [['x' for j in range(self.max_time_steps)] for i in
                                     range(gap)]
                Y_test_str_batch += [[self.dataManager.label2id['O'] for j in range(self.max_time_steps)] for i
                                     in range(gap)]
                y_test_psyduo_label_batch += [[self.dataManager.label2id['O'] for j in range(self.max_time_steps)]
                                              for i
                                              in range(gap)]
                X_test_batch = np.array(X_test_batch)
                X_test_str_batch = np.array(X_test_str_batch)
                Y_test_str_batch = np.array(Y_test_str_batch)
                y_test_psyduo_label_batch = np.array(y_test_psyduo_label_batch)
                results, token, real_entity, entity, real_entities_type, entities_type, _, _ = self.predictBatch(
                    self.sess, X_test_batch,
                    y_test_psyduo_label_batch,
                    X_test_str_batch, Y_test_str_batch)
                results = results[:len(X_test_batch)]
                token = token[:len(X_test_batch)]
                real_entity = real_entity[:len(X_test_batch)]
                entity = entity[:len(X_test_batch)]
                real_entities_type = real_entities_type[:len(X_test_batch)]
                entities_type = entities_type[:len(X_test_batch)]
            else:
                results, token, real_entity, entity, real_entities_type, entities_type, _, _ = self.predictBatch(
                    self.sess, X_test_batch,
                    y_test_psyduo_label_batch,
                    X_test_str_batch, Y_test_str_batch)

            labels.extend(results)
            tokens.extend(token)
            real_entities.extend(real_entity)
            entities.extend(entity)
            real_entities_types.extend(real_entities_type)
            entities_types.extend(entities_type)

        def save_test_out(tokens, labels):
            # transform format
            newtokens, newlabels = [], []
            for to, la in zip(tokens, labels):
                newtokens.extend(to)
                newtokens.append("")
                newlabels.extend(la)
                newlabels.append("")
            # save
            save_csv_(pd.DataFrame({"token": newtokens, "label": newlabels}), self.output_test_file,
                      ["token", "label"],
                      delimiter=self.configs.delimiter)

        save_test_out(tokens, labels)
        self.logger.info("testing results saved.\n")

        if self.is_output_sentence_entity:
            with open(self.output_sentence_entity_file, "w", encoding='utf-8') as outfile:
                # print('\033[1;31m pred entities is :  \033[0m', entities)
                # print('\033[1;31m the length of pred entities is :  \033[0m', len(entities))
                for i in range(len(entities)):
                    if self.configs.label_level == 1:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(entities[i]) + "\n\n")
                    elif self.configs.label_level == 2:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(
                            [a + "\t(%s)" % b for a, b in zip(entities[i], entities_types[i])]) + "\n\n")
                        # print(len(entities[i]))

            self.logger.info("testing results with sentences&entities saved.\n")

        if self.is_real_output_sentence_entity:
            with open(self.real_output_sentence_entity_file, "w", encoding='utf-8') as outfile:
                # print('\033[1;31m real entities is :  \033[0m', real_entities)
                # print('\033[1;31m the length of real entities is :  \033[0m', len(real_entities))
                for i in range(len(real_entities)):
                    if self.configs.label_level == 1:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(real_entities[i]) + "\n\n")
                    elif self.configs.label_level == 2:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(
                            [a + "\t(%s)" % b for a, b in zip(real_entities[i], real_entities_types[i])]) + "\n\n")
                        # print(len(real_entities[i]))

            self.logger.info("testing results with the real sentences&entities saved.\n")

        if self.is_compare_output_sentence_entity:
            with open(self.compare_output_sentence_entity_file, "w", encoding='utf-8') as outfile:
                for i in range(len(real_entities)):
                    if self.configs.label_level == 1:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(real_entities[i]) + "\n\n")
                    elif self.configs.label_level == 2:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(
                            [a + "\t(%s)" % b for a, b in
                             zip(real_entities[i], real_entities_types[i])]) + "\n\n" + "\n".join(
                            [c + "\t(%s)" % d for c, d in zip(entities[i], entities_types[i])]) + "\n\n")

            self.logger.info("testing results with the compare sentences&entities saved.\n")
        self.logger.info("Starting to calculate the correct_accurty and recall_accurty! \n")
        prd_list = list(chain(*entities))
        real_list = list(chain(*real_entities))
        correct_num = 0
        vocab_num = len(real_list)
        for i in range(len(prd_list)):
            if prd_list[i] in real_list:
                correct_num += 1
            else:
                vocab_num += 1
        correct_accurty = float(correct_num) / len(real_list)
        recall_accurty = float(correct_num) / vocab_num
        F = 2 * correct_accurty * recall_accurty / (correct_accurty + recall_accurty)
        self.logger.info("the correct accurty is : %.3f" % (correct_accurty))
        self.logger.info("the recall accurty is : %.3f" % (recall_accurty))
        self.logger.info("the F1 is : %.3f" % (F))

        self.sess.close()

    def predictBatch_real(self, sess, X, y_psydo_label, X_test_str_batch):
        entity_list = []
        tokens = []
        predicts_labels_entitylevel = []
        indexs = []
        predicts_labels_tokenlevel = []

        predicts_label_id, lengths = \
            sess.run([
                self.batch_pred_sequence,
                self.length
            ],
                feed_dict={
                    self.inputs: X,
                    self.targets: y_psydo_label,
                })

        for i in range(len(lengths)):
            x_ = [val for val in X_test_str_batch[i, 0:lengths[i]]]
            tokens.append(x_)

            y_pred = [str(self.dataManager.id2label[val]) for val in predicts_label_id[i, 0:lengths[i]]]
            predicts_labels_tokenlevel.append(y_pred)

            entitys, entity_labels, labled_indexs = extractEntity(x_, y_pred, self.dataManager)
            entity_list.append(entitys)
            predicts_labels_entitylevel.append(entity_labels)
            indexs.append(labled_indexs)

        return predicts_labels_tokenlevel, tokens, entity_list, predicts_labels_entitylevel, indexs

        # def predict_single(self, sentence):
        #     X, Sentence, Y = self.dataManager.prepare_single_sentence(sentence)
        #     _, tokens, entitys, predicts_labels_entitylevel, indexs = self.predictBatch(self.sess, X, Y, Sentence)
        #     return tokens[0], entitys[0], predicts_labels_entitylevel[0], indexs[0]

    def predictBatch(self, sess, X, y_psydo_label, X_test_str_batch, Y_test_str_batch):
        entity_list = []
        tokens = []
        predicts_labels_entitylevel = []
        indexs = []
        predicts_labels_tokenlevel = []
        real_entity_list = []
        real_labels_entitylevel = []
        real_indexs = []

        predicts_label_id, lengths = \
            sess.run([
                self.batch_pred_sequence,
                self.length
            ],
                feed_dict={
                    self.inputs: X,
                    self.targets: y_psydo_label,
                })
        # print(type(predicts_label_id))
        # print(type(y_psydo_label))
        # print(predicts_label_id)
        # print(y_psydo_label)

        for i in range(len(lengths)):
            x_ = [val for val in X_test_str_batch[i, 0:lengths[i]]]
            tokens.append(x_)
            y_ = [val for val in Y_test_str_batch[i, 0:lengths[i]]]
            # print(x_)

            y_pred = [str(self.dataManager.id2label[val]) for val in predicts_label_id[i, 0:lengths[i]]]
            # print(y_pred)
            predicts_labels_tokenlevel.append(y_pred)
            # print(y_)

            entitys, entity_labels, labled_indexs = extractEntity(x_, y_pred, self.dataManager)
            real_entitys, real_entity_labels, real_labled_indexs = extractEntity(x_, y_, self.dataManager)
            real_entity_list.append(real_entitys)
            entity_list.append(entitys)
            real_labels_entitylevel.append(real_entity_labels)
            predicts_labels_entitylevel.append(entity_labels)
            real_indexs.append(real_labled_indexs)
            indexs.append(labled_indexs)

        return predicts_labels_tokenlevel, tokens, real_entity_list, entity_list, real_labels_entitylevel, predicts_labels_entitylevel, real_indexs, indexs

    def soft_load(self):
        self.logger.info("loading model parameter")
        tf.initialize_all_variables().run(session=self.sess)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoints_dir))
        self.logger.info("loading model successfully")
