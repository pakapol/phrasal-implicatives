from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
import util
import tensorflow as tf
import os, glove
from model import PIModel
from configs.standard_conf import config
import numpy as np

flags = tf.flags
logging = tf.logging
#Check point paths are odd. You should have some folder where you saved your model. It should have 5 files in it of the form:
#checkpoint
#log
#xxx.data
#xxx.index
#xxx.meta
#The check point path you should provide as an argument is path/to/folder/xxx


flags.DEFINE_string("checkpoint_path", None, "checkpoint_path")
flags.DEFINE_string("config_path", None, "config_path")

FLAGS = flags.FLAGS

def run_sample(session, m, s1, s2, mask1, mask2, eval_op):
    """Runs the model on the given data."""
    pred, _ = session.run([m.logits, eval_op], 
                                      {m.prem_placeholder: [s1[0][0]],
                                          m.hyp_placeholder: [s2[0][0]],
                                          m.hyp_len_placeholder: [mask1],
                                          m.prem_len_placeholder: [mask2]})
    print (pred)
    return np.argmax(pred, axis=1)


def get_config(config_path):
    class conf(object): pass
    with open(config_path) as f:
        for line in f:
            if 'import' not in line and len(line) > 0:
                exec("conf."+line[:-1])
    return conf

def main(_):
    single_preset = config
    single_preset.batch_size = 1
    len_cap =single_preset.max_prem_len - 1
    word_to_id = glove._get_glove_vocab(r"glove/glove.6B.list", single_preset.vocab_limit)
    id_to_word = {}
    for word in word_to_id:
        id_to_word[word_to_id[word]] = word
    with tf.Session() as session:
        pretrained_embeddings = util._get_glove_vec("glove/glove.6B.300d.txt", vocab_limit=config.vocab_limit)
        m = PIModel(config, pretrained_embeddings)
        saver = tf.train.Saver()
        saver.restore(session, FLAGS.checkpoint_path)
        while 1:
            print('')
            in1 = str(input("Enter premise: ")).lower()
            in2 = str(input("Enter hypothesis: ")).lower()
            print('')
            s1 = np.array([util._sent_to_id(in1, word_to_id, len_cap+1)])
            s2 = np.array([util._sent_to_id(in2, word_to_id, len_cap+1)])
            pred = run_sample(session, m, s1, s2, len(in1.split()), len(in2.split()), tf.no_op())
            print(" ".join([id_to_word[ids] for ids in s1[0][0]][:len(in1.split())]))
            print(util._num_to_label(pred[0]))
            print(" ".join([id_to_word[ids] for ids in s2[0][0]][:len(in2.split())]))
	 
if __name__ == "__main__":
    tf.app.run()
