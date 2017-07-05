# verbose to stdout or log to file (in another format)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from configs.standard_conf import config
import reader, os
from model import PIModel
import numpy as np
from sklearn.metrics import f1_score
import util
import tensorflow as tf
import util
flags = tf.flags
logging = tf.logging
flags.DEFINE_string("checkpoint_path", None, "checkpoint_path")
flags.DEFINE_string("config_path", "conf.py", "config_path")

FLAGS = flags.FLAGS

def run_eval(session, m, data, eval_op):
    """Runs the model on the given data."""
    batch_size = 32
    costs = 0.0
    iters = 0
    totalacc = 0.0
    preds = []
    for prem, premlen, hyp,  hyplen, label, _ in data:
        logits, cost, _ = session.run([m.logits, m.loss, eval_op], 
                                      {m.prem_placeholder: prem,
                                          m.hyp_placeholder: hyp,
                                          m.hyp_len_placeholder: premlen,
                                          m.prem_len_placeholder: hyplen,
                                          m.label_placeholder: label})
        costs += cost
        iters += 1
        preds += list(np.argmax(logits, axis=1))
    return preds, costs / iters

def main(_):
    single_preset = config
    word_to_id = util._get_word_to_id("glove/glove.6B.list", vocab_limit=config.vocab_limit)
    valid_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='val', prefix='pi', shuffle=True)
    test_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='test', prefix='pi', shuffle=True)
    single_preset.batch_size = 1

    with tf.Graph().as_default(), tf.Session() as session:
        pretrained_embeddings = util._get_glove_vec("glove/glove.6B.300d.txt", vocab_limit=config.vocab_limit)
        m = PIModel(config, pretrained_embeddings)
        saver = tf.train.Saver()
        saver.restore(session, FLAGS.checkpoint_path)
        val_pred, valid_loss = run_eval(session, m, valid_data, tf.no_op())
        print("Val loss: %.3f\n" % (valid_loss))

        test_pred, test_loss = run_eval(session, m, test_data, tf.no_op())
        print("Test loss: %.3f\n" % (test_loss))

        with open(os.path.join(single_preset.data_path,'pred.val'),'w') as f:
            for p in val_pred:
                f.write(reader._revert(p) + '\n')
        with open(os.path.join(single_preset.data_path,'pred.test'), 'w') as g:
            for p in test_pred:
                g.write(reader._revert(p) + '\n')

if __name__ == "__main__":
    tf.app.run()
