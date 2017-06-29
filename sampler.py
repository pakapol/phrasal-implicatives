from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import reader, os, glove
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

def run_sample(session, s1, s2, mask1, mask2, eval_op):
    """Runs the model on the given data."""
    pred, _ = session.run([session.graph.get_operation_by_name("add_1").outputs[0], eval_op], # eliminated m.acc
                                      {"Placeholder:0": s1,
                                          "Placeholder_2:0": s2,
                                          "Placeholder_1:0": [mask1],
                                          "Placeholder_3:0": [mask2]})
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
    word_to_id = glove._get_glove_vocab(r"glove/glove.6B.list", single_preset.vocab_limit)
    id_to_word = {}
    for word in word_to_id:
        id_to_word[word_to_id[word]] = word
    with tf.Session() as session:
        len_cap = int(input("Enter Length Cap: "))
        saver = tf.train.import_meta_graph(FLAGS.checkpoint_path + ".meta")
        saver.restore(session, FLAGS.checkpoint_path)
        while 1:
            print('')
            in1 = str(input("Enter premise: ")).lower().split()
            in2 = str(input("Enter hypothesis: ")).lower().split()
            print('')
            s1 = np.array([reader._sentence_to_word_id((in1 + ["<eos>"] + ["<blank>"] * (len_cap - len(in1)))[:len_cap + 1], word_to_id)])
            s2 = np.array([reader._sentence_to_word_id((in2 + ["<eos>"] + ["<blank>"] * (len_cap - len(in2)))[:len_cap + 1], word_to_id)])
            pred = run_sample(session, s1, s2, len_cap, len_cap, tf.no_op())
            print(" ".join([id_to_word[ids] for ids in s1[0]][:len(in1)+1]))
            print(reader._revert(pred))
            print(" ".join([id_to_word[ids] for ids in s2[0]][:len(in2)+1]))
	 
if __name__ == "__main__":
    tf.app.run()
