from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections, itertools
import os, random

import numpy as np
import glove
from configs.standard_conf import config as conf

# Converting each label to numeric notation
def _convert(label):
    if label == 'entails':
        return 0
    elif label == 'contradicts':
        return 1
    elif label == 'permits':
        return 2
    else:
        raise Exception('LabelError: {}'.format(label))

def _revert(label):
    if label == 0:
        return 'entails'
    elif label == 1:
        return 'contradicts'
    elif label == 2:
        return 'permits'
    else:
        raise Exception('LabelError: {}'.format(label))

# Read premises and hypotheses
def _read_sent(filename):
    with open(filename, "r", encoding = "utf8") as f:
        lines = f.read().lower().split('\n')
        lines.remove("")
        sentences = [i.split() for i in lines]
        return sentences


def _read_prems(filename, len_cap=None):
    sentences = _read_sent(filename)
    sent_lens = [len(sent) for sent in sentences] # Equal to Index of word ("<eos>") that the mask will take value 1
    if len_cap is None:
        len_cap = max(sent_lens) + 1
    return [(sent + ["<eos>"] + ["<blank>"] * (len_cap - len(sent)))[:len_cap] for sent in sentences], sent_lens

def _read_hyps(filename, len_cap=None):
    sentences = _read_sent(filename)
    sent_lens = [len(sent) for sent in sentences] # Equal to Index of word ("<eos>") that the mask will take value 1
    if len_cap is None:
        len_cap = max(sent_lens) + 1
    return [(sent + ["<eos>"] + ["<blank>"] * (len_cap - len(sent)))[:len_cap] for sent in sentences], sent_lens

# Read in the labels: return a numerical
def _read_labels(filename):
    with open(filename, "r", encoding="utf8") as f:
        result = f.read().split('\n')
        result.remove("")
        return [_convert(label) for label in result]

# make vocabulary from premises and hypotheses (prems and hyps are lists)
def _get_vocab(prems, hyps):
    counter = collections.Counter(list(itertools.chain.from_iterable(prems + hyps)))
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    print("Vocab size: {}".format(len(words)))
    word_to_id = collections.defaultdict(lambda: 0, zip(words, range(2, len(words)+2)))
    word_to_id['<unk>'] = 0
    word_to_id['<blank>'] = 1
    return word_to_id

# convert prem lists and hyp lists to ids
def _sentences_to_word_ids(sentences, word_to_id):
    return [[word_to_id[word] for word in sent] for sent in sentences]

def _sentence_to_word_id(sentence, word_to_id):
    return [word_to_id[word] for word in sentence]

def pi_raw_data(max_len, data_path=None):
    """Load PI raw data from data directory "data_path".

    Reads PI text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PIiterator.
    """
    # Download the data: prem, hyp, label x train, val, test = 9 files
    prem_train_path = os.path.join(data_path, "pi.prem.train")
    hyp_train_path = os.path.join(data_path, "pi.hyp.train")
    label_train_path = os.path.join(data_path, "pi.label.train")

    prem_val_path = os.path.join(data_path, "pi.prem.val")
    hyp_val_path = os.path.join(data_path, "pi.hyp.val")
    label_val_path = os.path.join(data_path, "pi.label.val")

    prem_test_path = os.path.join(data_path, "pi.prem.test")
    hyp_test_path = os.path.join(data_path, "pi.hyp.test")
    label_test_path = os.path.join(data_path, "pi.label.test")

    # read train, val, test data
    prem_train, prem_train_len = _read_prems(prem_train_path, max_len)
    hyp_train, hyp_train_len = _read_hyps(hyp_train_path, max_len)
    label_train = _read_labels(label_train_path)

    prem_val, prem_val_len = _read_prems(prem_val_path, max_len) # originally has len_cap=max(prem_train_len)
    hyp_val, hyp_val_len = _read_hyps(hyp_val_path, max_len)
    label_val = _read_labels(label_val_path)

    prem_test, prem_test_len = _read_prems(prem_test_path, max_len)
    hyp_test, hyp_test_len = _read_hyps(hyp_test_path, max_len)
    label_test = _read_labels(label_test_path)

    word_to_id = glove._get_glove_vocab("glove/glove.6B.list", conf.vocab_limit)
    # word_to_id = _get_vocab(prem_train, hyp_train)
    train_data = (_sentences_to_word_ids(prem_train, word_to_id), _sentences_to_word_ids(hyp_train, word_to_id), prem_train_len, hyp_train_len, label_train)
    valid_data = (_sentences_to_word_ids(prem_val, word_to_id), _sentences_to_word_ids(hyp_val, word_to_id), prem_val_len, hyp_val_len, label_val)
    test_data = (_sentences_to_word_ids(prem_test, word_to_id), _sentences_to_word_ids(hyp_test, word_to_id), prem_test_len, hyp_test_len, label_test)

    return train_data, valid_data, test_data

def pi_iterator(raw_data, batch_size, reshuffle=True):
    """Iterate on the raw PTB data.

    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.

    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.

    Raises:
        ValueError: if batch_size or num_steps are too high.
    """

    prems, hyps, prem_len, hyp_len, labels = raw_data
    data_len = len(labels)

    if reshuffle:
        shuffler = list(range(data_len))
        random.shuffle(shuffler)

        prems = [prems[i] for i in shuffler]
        hyps = [hyps[i] for i in shuffler]
        prem_len = [prem_len[i] for i in shuffler]
        hyp_len = [hyp_len[i] for i in shuffler]
        labels = [labels[i] for i in shuffler]

    num_epoch = data_len // batch_size

    prems = np.array(prems, dtype=np.int32)
    premmasks = np.zeros((data_len, max(prem_len) + 1)) # need + 1 because of "<eos>"
    premmasks[np.arange(data_len), np.array(prem_len, dtype=np.int32)] = 1.0

    hyps = np.array(hyps, dtype=np.int32)
    hypmasks = np.zeros((data_len, max(hyp_len) + 1)) # need + 1 because of "<eos>"
    hypmasks[np.arange(data_len), np.array(hyp_len, dtype=np.int32)] = 1.0

    labels = np.array(labels, dtype=np.int32)

    for i in range(num_epoch):
        prem = prems[i * batch_size: (i+1) * batch_size]
        premlens = prem_len[i * batch_size: (i+1) * batch_size]

        hyp = hyps[i * batch_size: (i+1) * batch_size]
        hyplens = hyp_len[i * batch_size: (i+1) * batch_size]

        label = labels[i * batch_size: (i+1) * batch_size]

        yield (prem, hyp, premlens, hyplens, label)


def pi_remainder(raw_data, original_batch_size):
    prems, hyps, prem_len, hyp_len, labels = raw_data
    data_len = len(labels)

    num_remainder = data_len % original_batch_size

    prems = np.array(prems, dtype=np.int32)
    premmasks = np.zeros((data_len, max(prem_len) + 1)) # need + 1 because of "<eos>"
    premmasks[np.arange(data_len), np.array(prem_len, dtype=np.int32)] = 1.0

    hyps = np.array(hyps, dtype=np.int32)
    hypmasks = np.zeros((data_len, max(hyp_len) + 1)) # need + 1 because of "<eos>"
    hypmasks[np.arange(data_len), np.array(hyp_len, dtype=np.int32)] = 1.0

    labels = np.array(labels, dtype=np.int32)



    return (prems[-num_remainder:], hyps[-num_remainder:], premmasks[-num_remainder:], hypmasks[-num_remainder:], labels[-num_remainder:])
