# -*- coding: utf-8 -*-
# verbose to stdout or log to file (in another format)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reader, os
from model import PIModel
import numpy as np
import tensorflow as tf
import glove

test_data = {}
github_path = "/Users/laurik/eacl-2017-implicatives/"
data_path = "data0.10/0.80-block/"

class example:
    
    def __init__(self, prem, rel, hyp, pred):
        self.prem = prem
        self.rel = rel
        self.hyp = hyp
        self.pred = pred
        
    def __repr__(self):
        return 'prem = {}, rel= {}, hyp = {}, pred = {}'.format(self.prem,
        self.rel, self.hyp, self.pred) 
        
    def __str__(self):
        return '{}\n{} - {}\n{}\n'.format(self.prem, self.rel, self.pred,
        self.hyp)
        
probabilistic = ["be able", "be forced", "be prevented", "get chance", "have chance", "have time"]
counts = {"deterministic":0, "probabilistic":0}

constructions = ["be able", "be forced", "be prevented", "bother", "break pledge",
"break promise", "dare", "fail", "fail obligation", "follow order", "forget",
"fulfill promise", "get chance", "have chance", "have courage",
"have foresight", "have time", "hesitate", "keep promise", "lack foresight",
"lose opportunity","make promise", "make vow", "manage", "meet duty",
"meet obligation", "meet promise", "miss chance", "miss opportunity",
"neglect", "obey order", "remember", "take chance", "take no time",
"take opportunity", "take time", "take vow", "waste chance", "waste money",
"waste no time", "waste opportunity", "waste time"]

correct = {"be able" : 0, "be forced" : 0, "be prevented" : 0, "bother" : 0, "break pledge" : 0,
"break promise" : 0, "dare" : 0, "fail" : 0, "fail obligation" : 0, "follow order" : 0, "forget" : 0,
"fulfill promise" : 0, "get chance" : 0, "have chance" : 0, "have courage" : 0,
"have foresight" : 0, "have time" : 0, "hesitate" : 0, "keep promise" : 0, "lack foresight" : 0,
"lose opportunity" : 0,"make promise" : 0, "make vow" : 0, "manage" : 0, "meet duty" : 0,
"meet obligation" : 0, "meet promise" : 0, "miss chance" : 0, "miss opportunity" : 0,
"neglect" : 0, "obey order" : 0, "remember" : 0, "take chance" : 0, "take no time" : 0,
"take opportunity" : 0, "take time" : 0, "take vow" : 0, "waste chance" : 0, "waste money" : 0,
"waste no time" : 0, "waste opportunity" : 0, "waste time" : 0, "entails": 0, "permits":0,
"contradicts":0, "total" : 0}

error = {"be able" : 0, "be forced" : 0, "be prevented" : 0, "bother" : 0, "break pledge" : 0,
"break promise" : 0, "dare" : 0, "fail" : 0, "fail obligation" : 0, "follow order" : 0, "forget" : 0,
"fulfill promise" : 0, "get chance" : 0, "have chance" : 0, "have courage" : 0,
"have foresight" : 0, "have time" : 0, "hesitate" : 0, "keep promise" : 0, "lack foresight" : 0,
"lose opportunity" : 0,"make promise" : 0, "make vow" : 0, "manage" : 0, "meet duty" : 0,
"meet obligation" : 0, "meet promise" : 0, "miss chance" : 0, "miss opportunity" : 0,
"neglect" : 0, "obey order" : 0, "remember" : 0, "take chance" : 0, "take no time" : 0,
"take opportunity" : 0, "take time" : 0, "take vow" : 0, "waste chance" : 0, "waste money" : 0,
"waste no time" : 0, "waste opportunity" : 0, "waste time" : 0, "total" : 0, 
"entails" : {"entails": 0, "permits":0, "contradicts":0}, "permits" : {"entails":0, "permits": 0, "contradicts":0},
"contradicts" : {"entails": 0, "permits":0, "contradicts":0}}

examples = {"be able" : [], "be forced" : [], "be prevented" : [], "bother" : [], "break pledge" : [],
"break promise" : [], "dare" : [], "fail" : [], "fail obligation" : [], "follow order" : [], "forget" : [],
"fulfill promise" : [], "get chance" : [], "have chance" : [], "have courage" : [],
"have foresight" : [], "have time" : [], "hesitate" : [], "keep promise" : [], "lack foresight" : [],
"lose opportunity" : [],"make promise" : [], "make vow" : [], "manage" : [], "meet duty" : [],
"meet obligation" : [], "meet promise" : [], "miss chance" : [], "miss opportunity" : [],
"neglect" : [], "obey order" : [], "remember" : [], "take chance" : [], "take no time" : [],
"take opportunity" : [], "take time" : [], "take vow" : [], "waste chance" : [], "waste money" : [],
"waste no time" : [], "waste opportunity" : [], "waste time" : []}

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config_path", None, "config_path")

FLAGS = flags.FLAGS

def accuracy(e, c):
    return (1.0 - (float(e)/c)) * 100
    
def table_results_by_construction(constr_type, num_tests, path):
    n = 0
    ct = 0
    et = 0
    file = constr_type + "_results.tex"
    with open(path+file, "w") as f:
        if constr_type == "probabilistic":
            num = num_tests - counts["deterministic"]
            caption = "Probabilistic implicatives"
        else:
            num = num_tests - counts["probabilistic"]
            caption = "Deterministic implicatives"
        f.write("\\begin{table}[ht] \\label{" + constr_type + "_constructions}\n")
        f.write("\\begin{small}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("Construction  &  Correct & Error & Accuracy\\\\\n")
        f.write("\\hline\n")
        for item in constructions:
            if constr_type == "deterministic" and item in probabilistic:
                continue
            elif constr_type == "probabilistic" and item not in probabilistic:
                continue
            n += 1
            ct += correct[item]
            et += error[item]
            item_acc = accuracy(error[item], correct[item])
            f.write("{} & {} & {} & {:.2f}\\%\\\\\n".format(item, correct[item], error[item], item_acc))           
        #ct = correct["total"]
        #et = error["total"]
        st = ct+et
        at = accuracy(et, ct)
        f.write("\\hline\\hline\n")
        f.write("{} constructions, & {} & {} & {:.2f}\\%\\\\\n".format(n, ct, et, at))
        f.write("\\end{tabular}\n")
        f.write("\\begin{tabular}{l}\n")
        f.write("{} tests\\\\\n".format(num))
        f.write("\\end{tabular}\n")
        f.write("\\caption {" + caption + "}\n")
        f.write("\\end{small}\n")
        f.write("\\end{table}\n")

        
def print_confusion_matrix(path):
    with open(path+"confusion_matrix.tex", "w") as f:
        totals = {"entails":0, "permits":0, "contradicts":0}
        f.write("\\begin{table}[ht] \label{confusion_matrix}\n")
        f.write("\\begin{small}\n")
        f.write("\\begin{tabular}{lccc}\n")
        #f.write("Correct &  & & Incorrect predictions\\\\\n")
        f.write("Correct &  & & Model predictions\\\\\n")                                                                                                                                                                                                       
        f.write("\\end{tabular}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\centering\n")
        f.write("relation    &     entails  &  permits & contradicts & Totals\\\\\n")
        f.write("\\hline\n")
        for expected in ["entails", "permits", "contradicts"]:
            row = error[expected]
            # if there is no confusion, we report the number of correct predictions
            if row[expected] == 0:
                row[expected] = correct[expected]
            sum = row["entails"] + row["permits"] + row["contradicts"]
            for column in ["entails", "permits", "contradicts"]:
                totals[column] += row[column]
            f.write("{} & {} & {}  & {} & {}\\\\\n".format(expected, row["entails"], row["permits"],row["contradicts"],sum))
        ts = totals["entails"] + totals["permits"] + totals["contradicts"]
        f.write("\\hline \\hline\n")
        f.write("Totals: & {} & {}  & {} & {}\\\\\n".format(totals["entails"], totals["permits"], totals["contradicts"], ts))
        f.write("\\end{tabular}\n")
        f.write("\\caption {Confusion matrix}\n")
        f.write("\\end{small}\n")
        f.write("\end{table}\n")

def print_test_data(path, num_tests):
    with open(path, "w") as f:
        num_constrs = len(constructions)
        f.write("{} Test Examples of {} constructions in {}\n".format(num_tests, num_constrs, data_path))
        f.write("====================================================================\n\n")
        for constr in constructions:
            exampls = examples[constr]
            n = len(exampls)
            corr = correct[constr]
            err = error[constr]
            f.write("Construction: {} - {} examples, correct: {}, errors: {}\n\n".format(constr, n, corr, err))
            f.write("---------------------------------------------------------------------\n\n")
            for ex in exampls:
                ex_str = ex.__str__()
                f.write("{}\n".format(ex_str))
            f.write("=====================================================================\n\n")


def run_eval(session, m, data, eval_op):   
    with open(data_path + "pi.prem.test") as g, open(data_path + "pi.hyp.test") as f, open(data_path + "pi.label.test") as h, open(data_path + "pi.constr.test") as e:
        prems = g.read().split('\n')
        hyps = f.read().split('\n')
        labels = h.read().split('\n')
        constructions = e.read().split('\n')
    
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
        lab = reader._revert(pred[0])
        #print("{}: {}\n{} - {} - {}\n{}\n".format(constructions[iters], prems[iters], labels[iters], lab, acc, hyps[iters]))
        constr = constructions[iters]
        exampls = examples[constr]
        exampls.append(example(prems[iters], labels[iters], hyps[iters], lab))
        if constr in probabilistic:
            counts["probabilistic"] += 1
        else:
            counts["deterministic"] += 1
        if acc == 1.0:
            correct[constr] += 1
            correct[lab] += 1
            correct["total"] += 1
        elif acc == 0.0:
            error[constr] += 1
            error["total"] += 1
            exp = labels[iters]
            cell = error[exp]
            cell[lab] += 1            
        else:
            print("Weird error. acc = {}".format(acc))
        costs += cost
        totalacc += acc
        iters += 1
        preds += list(pred)
    return preds, costs / iters, totalacc / iters, correct, error, iters

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
        tf.initialize_all_variables().run()

        # Retrieving model checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(single_preset.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No checkpoint found. Set a valid --checkpoint_path for model evaluation")

        # m.assign_lr(session, batch_preset.learning_rate)

        #val_pred, valid_loss, valid_acc = run_eval(session, m_val, valid_data, tf.no_op())
        #print("Val loss: %.3f, acc: %.3f\n" % (valid_loss, valid_acc))

        test_pred, test_loss, test_acc, correct, error, num_tests = run_eval(session, m_test, test_data, tf.no_op())
        print("Test loss: %.3f, acc: %.3f\n" % (test_loss, test_acc))
        
        # print separate tables for deterministic and probabilistic constructions
        table_results_by_construction("deterministic", num_tests, github_path)
        table_results_by_construction("probabilistic", num_tests, github_path)
        print_confusion_matrix(github_path)
        print_test_data("test_data.txt", num_tests)
        print_test_data(github_path +"test_data.txt", num_tests)
        
        #    for p in val_pred:
        #        f.write(reader._revert(p) + '\n')
        #with open(os.path.join(single_preset.data_path,'pred.test'), 'w') as g:
        #    for p in test_pred:
        #        g.write(reader._revert(p) + '\n')

if __name__ == "__main__":
    tf.app.run()
