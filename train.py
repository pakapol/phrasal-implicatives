from __future__ import absolute_import, division, print_function

import sys, os, time, random, collections, itertools, importlib
import util
import numpy as np
import tensorflow as tf
from model import PIModel

def main(_):
    config = importlib.import_module("configs.{}".format(sys.argv[1])).config # exec("from configs.{} import config".format(sys.argv[1]))
    ## get embedding
    pretrained_embeddings = util._get_glove_vec("glove/glove.6B.300d.txt", vocab_limit=config.vocab_limit)
    word_to_id = util._get_word_to_id("glove/glove.6B.list", vocab_limit=config.vocab_limit)

    ## Initialize model and tf session

    m = PIModel(config, pretrained_embeddings)
    labels = ['entails','contradicts','permits']
    cat_names = ['{}=>{}'.format(x,y) for x,y in itertools.product(labels,labels)]
    logs = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ## Iterate
        for ep in range(config.num_epoch):
            print("Begin epoch {}".format(ep))
            train_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='train', prefix='pi', shuffle=True)
            preds_t, labels_t, constr_t, loss_t = m.run_train_epoch(sess, train_data)
            val_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='val', prefix='pi', shuffle=False)
            preds_v, labels_v, constr_v, loss_v = m.run_val_epoch(sess, val_data)
            
            s0 = util.get_evaluation(preds_t, labels_t, metric='f1_macro', by_construct=False, constr=constr_t)
            s1 = util.get_evaluation(preds_t, labels_t, metric='f1_micro', by_construct=False, constr=constr_t)
            s2 = util.get_evaluation(preds_v, labels_v, metric='f1_macro', by_construct=False, constr=constr_v)
            s3 = util.get_evaluation(preds_v, labels_v, metric='f1_micro', by_construct=False, constr=constr_v)
            
            s_cat = util.get_evaluation(preds_v, labels_v, metric='confusion_matrix', by_construct=False, constr=constr_v)
            #for constr in s_cat:
            #    logs[constr].append(s_cat[constr])
            print("Train loss = {} :: Val loss = {}".format(loss_t, loss_v))
            print("Train F1 macro/micro = {}/{} :: Val F1 macro/micro = {}/{}".format(s0, s1, s2, s3))
            mat = s_cat / np.sum(s_cat, axis=1, keepdims=True)
            print(mat.reshape(9))
            logs.append(mat.reshape(9))
            print("End of epoch {}\n".format(ep))

        time_train_done = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        dest_dir = os.path.join("results", "{}-{}".format(sys.argv[1], time_train_done))
        os.makedirs(dest_dir)    
    
        tf.train.Saver().save(sess, os.path.join(dest_dir, 'model'))

    with open(os.path.join(dest_dir,'log'), 'w') as f:
        for cat_name, scores in zip(cat_names, zip(*logs)):
            f.write(cat_name + " " + " ".join(list(map(str, scores))) + "\n")


if __name__ == '__main__':
    tf.app.run()
