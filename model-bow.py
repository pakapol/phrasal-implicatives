from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

class PIModel(object):
    def __init__(self, config, pretrained_embeddings):
        self.config = config
        self.embeddings = tf.Variable(pretrained_embeddings, trainable=self.config.retrain_embeddings)
        self.add_placeholders()
        self.add_embeddings()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_train_op()

    def add_placeholders(self):
        self.prem_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_prem_len))
        self.prem_len_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.hyp_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_hyp_len))
        self.hyp_len_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None,))

    def create_feed_dict(self, prem_batch, prem_len, hyp_batch, hyp_len, label_batch=None):
        feed_dict = {
            self.prem_placeholder: prem_batch,
            self.prem_len_placeholder: prem_len,
            self.hyp_placeholder: hyp_batch,
            self.hyp_len_placeholder: hyp_len,
        }
        if label_batch is not None:
            feed_dict[self.label_placeholder] = label_batch
        return feed_dict

    def add_embeddings(self):
        self.embed_prems = tf.nn.embedding_lookup(self.embeddings, self.prem_placeholder)
        self.embed_hyps = tf.nn.embedding_lookup(self.embeddings, self.hyp_placeholder)

    def add_prediction_op(self):
        
        initer = tf.contrib.layers.xavier_initializer()

        prems_bag = tf.reduce_sum(self.embed_prems, axis=1) / tf.to_float(tf.expand_dims(self.prem_len_placeholder, 1)) # -1 x time_step x vocab_size
        hyps_bag = tf.reduce_sum(self.embed_hyps, axis=1) / tf.to_float(tf.expand_dims(self.hyp_len_placeholder, 1))
        
        # vocab_size => 150 and concat
        P1 = tf.Variable(initer([300,150]))
        P2 = tf.Variable(initer([300,150]))
        feat = tf.concat([tf.matmul(prems_bag, P1), tf.matmul(hyps_bag, P2)], axis=1)
        # Feed-forward
        W1 = tf.Variable(initer([300,300]))
        W2 = tf.Variable(initer([300,300]))
        W3 = tf.Variable(initer([300,300]))
        b1 = tf.Variable(tf.zeros([1,300]) + 1e-3)
        b2 = tf.Variable(tf.zeros([1,300]) + 1e-3)
        b3 = tf.Variable(tf.zeros([1,300]) + 1e-3)
        H1 = tf.tanh(tf.matmul(feat, W1) + b1)
        H2 = tf.tanh(tf.matmul(H1, W2) + b2)
        outputs = tf.tanh(tf.matmul(H2, W3) + b3)
        # softmax
        
        Ws = tf.Variable(initer([300,3]))
        bs = tf.Variable(tf.zeros([1,3]) + 1e-3)

        self.logits = tf.matmul(outputs, Ws) + bs

    def add_loss_op(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.logits))

    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer()
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def optimize(self, sess, train_x, train_y):
        prem_batch, prem_len, hyp_batch, hyp_len = train_x
        label_batch = train_y
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, label_batch)
        output_feed = [self.train_op, self.logits, self.loss]
        _, logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def validate(self, sess, valid_x, valid_y):
        prem_batch, prem_len, hyp_batch, hyp_len = valid_x
        label_batch = valid_y
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, label_batch)
        output_feed = [self.logits, self.loss]
        logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def predict(self, sess, test_x):
        prem_batch, prem_len, hyp_batch, hyp_len = test_x
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len)
        output_feed = [self.logits]
        logits = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1)

    def run_train_epoch(self, sess, dataset):
        preds = []
        labels = []
        constrs = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label, constr in dataset:
            pred, loss = self.optimize(sess, (prem, prem_len, hyp, hyp_len), label)
            preds.extend(pred)
            labels.extend(label)
            constrs.extend(constr)
            losses += loss * len(label)
        return preds, labels, constrs, losses / len(labels)

    def run_val_epoch(self, sess, dataset):
        preds = []
        labels = []
        constrs = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label, constr in dataset:
            pred, loss = self.validate(sess, (prem, prem_len, hyp, hyp_len), label)
            preds.extend(pred)
            labels.extend(label)
            constrs.extend(constr)
            losses += loss * len(label)
        return preds, labels, constrs, losses / len(labels)

    def run_test_epoch(self, sess, dataset):
        preds = []
        labels = []
        constrs = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label, constr in dataset:
            pred = self.predict(sess, (prem, prem_len, hyp, hyp_len))
            preds.extend(pred)
            labels.extend(label)
            constrs.extend(constr)
        return preds, labels, constrs
