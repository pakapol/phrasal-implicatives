from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import reader, os, glove
from model import PIModel

import numpy as np

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("checkpoint_path", None, "checkpoint_path")
flags.DEFINE_string("config_path", None, "config_path")

FLAGS = flags.FLAGS

def run_sample(session, m, s1, s2, mask1, mask2, eval_op):
    """Runs the model on the given data."""
    
    pred, _ = session.run([m.pred, eval_op], # eliminated m.acc
                                     {m.input_prem: s1,
                                      m.input_hyp: s2,
                                      m.prem_mask: mask1,
                                      m.hyp_mask: mask2,
                                      m.targets: np.array([0])})
    return pred


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
    single_preset.batch_size = 1

    with tf.Graph().as_default(), tf.Session() as session:
        word_to_id = glove._get_glove_vocab("glove.6B.list", single_preset.vocab_size)
        id_to_word = glove._get_glove_lookup("glove.6B.list", single_preset.vocab_size)
        initializer = tf.random_normal_initializer(mean=0.0,
                                                   stddev=single_preset.init_scale)
        len_cap = int(raw_input("Enter Length Cap: "))
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PIModel(is_training=False, num_steps_prem=len_cap+1,
                             num_steps_hyp=len_cap+1, preset=single_preset)

        tf.global_variables_initializer().run()

        # Retrieving model checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No checkpoint found. Set a valid --checkpoint_path for model evaluation")
        while 1:
            print('')
            in1 = str(raw_input("Enter premise: ")).lower().split()
            in2 = str(raw_input("Enter hypothesis: ")).lower().split()
            print('')
            s1 = np.array([reader._sentence_to_word_id((in1 + ["<eos>"] + ["<blank>"] * (len_cap - len(in1)))[:len_cap + 1], word_to_id)])
            s2 = np.array([reader._sentence_to_word_id((in2 + ["<eos>"] + ["<blank>"] * (len_cap - len(in2)))[:len_cap + 1], word_to_id)])
            mask1 = np.zeros((1, len_cap + 1))
            mask2 = np.zeros((1, len_cap + 1))
            mask1[0, len(in1)] = 1.0
            mask2[0, len(in2)] = 1.0
            pred = run_sample(session, m, s1, s2, mask1, mask2, tf.no_op())
            print(" ".join([id_to_word[ids] for ids in s1[0]][:len(in1)+1]))
            print(reader._revert(pred))
            print(" ".join([id_to_word[ids] for ids in s2[0]][:len(in2)+1]))
	 
if __name__ == "__main__":
    tf.app.run()
