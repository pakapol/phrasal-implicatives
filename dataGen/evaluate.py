# verbose to stdout or log to file (in another format)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reader, os
from model import PIModel
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config_path", "conf.py", "config_path")

FLAGS = flags.FLAGS

def run_eval(session, m, data, eval_op):
    """Runs the model on the given data."""
    num_epoch = len(data[0]) // m.batch_size
    costs = 0.0
    iters = 0
    totalacc = 0.0
    preds = []
    for step, (prem, hyp, premmask, hypmask, label) in enumerate(reader.pi_iterator(data, m.batch_size, reshuffle=False)):
        pred, cost, state, acc, _ = session.run([m.pred, m.cost, m.final_state, m.acc, eval_op], # eliminated m.acc
                                     {m.input_prem: prem,
                                      m.input_hyp: hyp,
                                      m.prem_mask: premmask,
                                      m.hyp_mask: hypmask,
                                      m.targets: label})
        costs += cost
        totalacc += acc
        iters += 1
        preds += list(pred)
    return preds, costs / iters, totalacc / iters

def get_config(config_path):
    class conf(object): pass
    with open(config_path) as f:
        for line in f:
            if 'import' not in line and len(line) > 0:
                exec("conf."+line[:-1])
    return conf

def main(_):
    if not FLAGS.config_path:
        raise ValueError("Must set --config_path")

    single_preset = get_config(FLAGS.config_path)
    raw_data = reader.pi_raw_data(single_preset.data_path)
    train_data, valid_data, test_data = raw_data

    single_preset.batch_size = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_normal_initializer(mean=0.0,
                                                   stddev=single_preset.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m_val = PIModel(is_training=False, num_steps_prem=max(valid_data[2])+1,
                             num_steps_hyp=max(valid_data[3])+1, preset=single_preset)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            m_test = PIModel(is_training=False, num_steps_prem=max(test_data[2])+1,
                            num_steps_hyp=max(test_data[3])+1, preset=single_preset)


        tf.global_variables_initializer().run()

        # Retrieving model checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(single_preset.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No checkpoint found. Set a valid --checkpoint_path for model evaluation")

        # m.assign_lr(session, batch_preset.learning_rate)

        val_pred, valid_loss, valid_acc = run_eval(session, m_val, valid_data, tf.no_op())
        print("Val loss: %.3f, acc: %.3f\n" % (valid_loss, valid_acc))

        test_pred, test_loss, test_acc = run_eval(session, m_test, test_data, tf.no_op())
        print("Test loss: %.3f, acc: %.3f\n" % (test_loss, test_acc))

        with open(os.path.join(single_preset.data_path,'pred.val'),'w') as f:
            for p in val_pred:
                f.write(reader._revert(p) + '\n')
        with open(os.path.join(single_preset.data_path,'pred.test'), 'w') as g:
            for p in test_pred:
                g.write(reader._revert(p) + '\n')

if __name__ == "__main__":
    tf.app.run()
