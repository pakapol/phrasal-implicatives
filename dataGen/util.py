# -*- coding: utf-8 -*-
from pattern.en import conjugate
from copy import deepcopy
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import formdict
import re
from random import randint

stats = {}

def ANDOR(**kwargs):
    return "or" if kwargs["neg"] else "and"
    
def ANDNOR(**kwargs):
    return "nor" if kwargs["neg"] else "and"

def SubjPro(**kwargs):
    return formdict.subjPro[kwargs["PERSON"]]

def ObjPro(**kwargs):
    return formdict.objPro[kwargs["PERSON"]]

def PossPro(**kwargs):
    return formdict.possPro[kwargs["PERSON"]]
    
def ReflPro(**kwargs):
    return formdict.reflPro[kwargs["PERSON"]]
    
def ResPro(**kwargs):
    return "each other"

funcdict = {
    "AND,OR": ANDOR,
    "AND,NOR": ANDNOR, 
    "SubjPro": SubjPro,
    "ObjPro": ObjPro,
    "PossPro": PossPro,
    "ReflPro": ReflPro,
    "ResPro": ResPro,
}

negatives = ['not', 'never', 'almost', "n't"]

def choose_negForm():
    #pick one randomly
    return negatives[randint(0,3)]

def _call(fname, **kwargs):
    return funcdict[fname](**kwargs)

def checkvalidity(inputstring):
    pass

def getAllSentences(inputstring):
    checkvalidity(inputstring)
    tokens = inputstring.split()
    tokenizedList = [[]]
    inparen = ""
    for t in tokens:
        if re.match('^\([\w\,~]+\)$', t):
            appender = deepcopy(tokenizedList)
            for opt in t[1:-1].split(','):
                # ~ is a placeholder for a missing article, no need to include
                # in the implicative signature key such as miss (a,the,~,PossPro)
                if opt != u'~' :
                    tokenizedList += [x + [opt] for x in appender]
        elif re.match('^\([\w\,]+$', t):
            inparen += t[1:] + " "
        elif re.match('^[\w\,]+\)$', t):
            inparen += t[:-1]
            commasplit = inparen.split(',')
            toadd = [x.split() for x in commasplit]
            appender = deepcopy(tokenizedList)
            for optphrase in toadd:
                tokenizedList += [x + optphrase for x in appender]
            inparen = ""
        elif len(inparen) > 1:
            inparen += t + " "
        else:
            for current in tokenizedList:
                current.append(t)
    return tokenizedList

def simpleParse(sent):
    # Splits the premise sentence into three parts: the subject phrase, the
    # implicative phrase and the VP phrase. For example:
    # [[u'Ren\xe9e', u'F,S3'], ['$V', u'got', u'chance'], [u'to', '$V', u'reply']]
    fragments = [[],[],[]]
    count = 0
    for i in xrange(len(sent)):
        m = re.match("^([\w\,.'-]+)\[([\w\,]+)\]$",sent[i], re.UNICODE)
        if m:
            labels = tuple(m.group(2).split(','))
            if count == 0:
                if 'V' in labels or 'v' in labels:
                    raise Exception('ParseError: Missing subject NP label in \"{}\"'.format(' '.join(sent)))
                elif count < 3:
                    fragments[count].append(m.group(1))
                    fragments[count].append(','.join(labels))
                    count += 1
                else:
                    raise Exception('LabelError: Too many verbs in \"{}\"'.format(sent))
            elif 'V' in labels or 'v' in labels:
                fragments[count].append("$V")
                fragments[count].append(m.group(1)) #TODO: normalize verb
            else:
                fragments[count].append(sent[i])
        else:
            if sent[i] == 'to' and count == 1:
                if '$V' not in fragments[count]:
                    raise Exception('LabelError: Missing a [V] label in \"{}\"'.format(fragments[count]))
                count += 1
            fragments[count].append(sent[i])
    if '$V' not in fragments[2]:
        Exception('LabelError: Missing a [V] label in \"{}\"'.format(fragments[count]))
    return tuple(fragments)

def initialVowel(word):
    ltr = word[0]
    if ltr == u'a' or ltr == u'a' or ltr == u'e' or ltr ==  u'i' or ltr ==  u'o':
        return True
    else:
        return False


def toPremise(**kwargs):
    NP = kwargs["NP"]
    IMP = kwargs["IMP"]
    VP = kwargs["VP"]
    PERSON = kwargs["PERSON"]
    tense = kwargs["tense"]
    neg = kwargs["neg"]
    left = []
    newIMP = []
    verbtag = 0
    negForm = 'Choose'
    # if negation is expessed in IMP as in 'had no chance' we do not
    # change 'had' to 'did not have'
    if u'no' in IMP and neg == True:
        didNot = False
    else:
        didNot = neg
    for i in xrange(len(IMP)):
        if IMP[i] == '$V':
            verbtag += 1
        elif verbtag == 1:
                left, right, negForm = convertTense(IMP[i], tense, PERSON, didNot, negForm)
                newIMP += right
                verbtag -= 1
        elif IMP[i] == '[PossPro]':
            newIMP.append(formdict.possPro[PERSON])
        else:
            if IMP[i] == u'no' and neg == False:
                if IMP[i+1] == u'time':
                    newIMP.append(u'some')
                else:
                    if initialVowel(IMP[-1]):
                        newIMP.append(u'an')
                    else:
                        newIMP.append(u'a')
            else:
                newIMP.append(IMP[i])
    newVP = []
    for w in VP:
        if re.match('^\[([\w\,]+)\]$', w):
            newVP.append(_call(re.match('^\[([\w\,]+)\]$', w).group(1), **kwargs))
        elif w[0] != '$':
            newVP.append(w)
    return NP + left + newIMP + newVP, negForm

def toHypothesis(**kwargs):
    NP = kwargs["NP"]
    VP = kwargs["VP"][1:]
    left = []
    PERSON = kwargs["PERSON"]
    tense = kwargs["tense"]
    neg = kwargs["neg"]
    negForm = kwargs["negForm"]
    newVP = []
    verbtag = 0
    for i in xrange(len(VP)):
        if VP[i] == '$V':
            verbtag += 1
        elif verbtag == 1:
                left, right, negForm = convertTense(VP[i], tense, PERSON, neg, negForm)
                newVP += right
                verbtag -= 1
        elif re.match('^\[([\w\,]+)\]$', VP[i]):
            newVP.append( _call(re.match('^\[([\w\,]+)\]$', VP[i]).group(1), **kwargs))
        else:
            newVP.append(VP[i])
    return NP + left + newVP

def convertTense(w, tense, PERSON, neg=False, premForm=None):
    past, perfect = tense.split('.')
    # if neg is True and premForm is eiher 'almost' or 'never', the hypothesis can be
    # negated with the same form or with["not", "n't"]. If the premise is
    # negated with "['not' or "n't"] the hypothesis must be negated with  "not"
    # or "n't" because 'almost' and 'never' would add information not present in
    # the premise.
    # If negForm is 'almost' or 'never' it can precede or follow the auxiliary
    # that is, 'has never' and 'never has' are both OK but we prefer the former
    # 90% of time. In simple past we prefer 'never took' to 'did never take'.
    if neg:
        if premForm == 'Choose':
            negForm = choose_negForm()
        elif premForm in ['almost', 'never']:
            negForm = [premForm, "not", "n't"][randint(0,2)]
        elif randint(0,1):
            negForm = "n't"
        else:
            negForm = "not"
    else:
        negForm = None
    if perfect == 'Perfect':
        tohave = 'has' if 'S3' in PERSON else 'have'
        if past =='Past':
            tohave = 'had'
        if not neg:
            return ([tohave], [enform(w)], negForm)
        elif negForm in ['almost', 'never'] and randint(1,10) > 9:
            return ([negForm, tohave], [enform(w)], negForm)
        else:
            return ([tohave, negForm], [enform(w)], negForm)
    elif past == 'Past':
        if w in ['be','is','am','are','was','were','been']:
            tobe = 'was' if 'S3' in PERSON or 'S1' in PERSON else 'were'
            if not neg:
                return ([tobe], [], negForm)
            elif negForm in ['almost', 'never'] and randint(1,10) > 9:
                return ([negForm, tobe], [], negForm)
            else:
                return ([tobe, negForm], [], negForm)
        else:
            if not neg:
                return ([],[pastform(w)], negForm)
            elif negForm in ['almost', 'never']:
                return ([negForm], [conjugate(w, tense='past')], negForm)
            else:
                return (['did', negForm],[conjugate(w, tense='infinitive')], negForm)
    elif 'S3' in PERSON:
        if w in ['be','is','am','are','was','were','been']:
            tobe = 'is' if 'S3' in PERSON else 'am' if 'S1' in PERSON else 'are'
            return ([tobe], [], negForm) if not neg else ([tobe, negForm], [], negForm)
        else:
            return ([],[sgform(w)]) if not neg else (['does', negForm],[w], negForm)
    elif w in ['be','is','am','are','was','were','been']:
            tobe = 'am' if 'S1' in PERSON else 'are'
            if not neg:
                return ([tobe], [], negForm)
            elif negForm in ['almost', 'never'] and randint(1,10) > 9:
               return ([negForm, tobe], [], negForm)
            else:
               return ([tobe, negForm], [], negForm) 
    else:
        if not neg:
            return [w]
        elif negForm in ['almost', 'never']:
            return ([negForm, w], [], negForm)
        else:
            return (['do', negForm], [w], negForm)

def enform(w):
    result = conjugate(w, tense='pastparticiple')
    if result == 'leaved':
        return u'left'
    elif result == 'payed':
        return u'paid'
    elif result == u'curst':
        return u'cursed'
    else:        
        return result

def pastform(w):
    result = conjugate(w, tense='past')
    if result == 'leaved':
        return u'left'
    elif result == 'payed':
        return u'paid'
    elif result == u'curst':
        return u'cursed'
    else:        
        return result

def sgform(w):
    return conjugate(w, tense='present', person=3)

def ingform(w):
    return conjugate(w, tense='participle')

def bareform(w):
    #fixing errors in pattern.en
    if w == u'left':
        return u'leave'
    elif w == u'paid':
        return u'pay'
    elif w == u'used':
        return u'use'
    else:
        return conjugate(w, tense='infinitive')
    
def backToString(wrdList):
    if len(wrdList) > 0:
        result = wrdList[0]
        for item in wrdList[1:]:
            if item != u'~':
                result = result + ' ' + item
        return result
    else:
        return u''

def lookupPI(IMP, *dicts):
    cleanedIMPtoken = []
    verbtag = 0
    for i in xrange(len(IMP)):
        if IMP[i] == '$V':
            verbtag += 1
        elif verbtag == 1:
            cleanedIMPtoken.append(bareform(IMP[i]))
            verbtag -= 1
        else:
            cleanedIMPtoken.append(IMP[i])
    constr = tuple(cleanedIMPtoken)
    
    for dict in dicts:
        if constr in dict:
            return constr, SignToProb(dict[constr])
    else:
        maxSub = 0
        prob = None
        for dict in dicts:
            for x in dict:
                if subsequence(x,cleanedIMPtoken) > maxSub:
                    prob = SignToProb(dict[x])
        if prob is None:
            badIMP = backToString(cleanedIMPtoken)
            raise Exception("No implicatives for {} in the dictionaries provided".format(badIMP))
        else:
            return constr, prob

# use subsequence look-up

def subsequence(list1,list2):
    curr = 0
    for i in xrange(len(list1)):
        if list1[i] not in list2[curr:]:
            return 0
        else:
            curr += list2[curr:].index(list1[i]) + 1
    return len(list1)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def SignToProb(sign):
    left,right = sign.split('|')
    if left == '+':
        newleft = 1.0
    elif left == '-':
        newleft = -1.0
    elif left == 'o':
        newleft = left
    elif left[0] in ['+','-']:
        if isfloat(left):
            newleft = float(left)
        else:
            raise Exception("SignError: {}".format(sign))
    elif left[-1] in ['+','-']:
        if isfloat(left[:-1]):
            newleft = float(left[:-1]) * (-1 if left[-1] == '-' else 1)
        else:
            raise Exception("SignError: {}".format(sign))
    elif isfloat(left):
        if '+' in right:
            newleft = -float(left)
        else:
            newleft = float(left)

    if right == '+':
        newright = 1.0
    elif right == '-':
        newright = -1.0
    elif right == 'o':
        newright = right
    elif right[0] in ['+','-']:
        if isfloat(right):
            newright = float(right)
        else:
            raise Exception("SignError: {}".format(sign))
    elif right[-1] in ['+','-']:
        if isfloat(right[:-1]):
            newright = float(right[:-1]) * (-1 if right[-1] == '-' else 1)
        else:
            raise Exception("SignError: {}".format(sign))
    elif isfloat(right):
        if '-' in left:
            newright = float(right)
        else:
            newright = -abs(float(right))
    return newleft, newright
    
def getFiles(directory):
    # get data files from the directory and prune unwanted
    # *_temp.txt files
    namelist = os.listdir(directory)
    if '.DS_Store' in namelist:
        namelist.remove('.DS_Store')
    for name in namelist:
        if name.rfind('_temp.txt') > 0:
            namelist.remove(name)
            os.remove(directory+'/'+name)
    return namelist
        
class stats:

    def __init__(self):
        self.counts = {'premLen':{}, 'hypLen':{}, 'exampleLen':{}, 'prem-hypLen':{},
        'entails':0, 'contradicts':0, 'permits':0}
        self.plot = None
        
    def log(self, g, line):
        sys.stdout.write(line)
        g.write(line)
        
    def latex(self, f, line):
        f.write(line)
        
    def latex_header(self, f):
        f.write("\\begin{table}[t]\n")
        f.write("\\begin{small}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{tabular}{lcr} \label{tb:constructions}\n")
        f.write("Construction  & Sign & Examples\\\\\n\hline\n")
        
    def latex_close(self, f, num_constr, num_examples):
        counts = self.counts
        f.write("\hline \hline \n")
        f.write("{} constructions & Total & {}\n".format(num_constr, num_examples))
        f.write("\\end{tabular}\n")
        f.write("\\end{center}\n")
        f.write("\\begin{tabular}{ccc}\n")
        f.write("entails: {} & contradicts: {} & permits: {}\\\\\n\n".format(
        counts['entails'], counts['contradicts'], counts['permits']))
        f.write("\\end{tabular}\n")
        f.write("\\caption{Implicative Constructions in \imc}\n")
        f.write("\\end{small}\n")
        f.write("\\end{table}\n")
        
      
    def updateStats(self, attr,  len):
        cell = self.counts[attr]
        if cell.has_key(len):
            cell[len] = cell[len]+1
        else:
            cell[len] = 1
        return len
        
    def addToRelation(self, label):
        self.counts[label] = self.counts[label]+1
    
    def completeDict(self, dict):
        # id the 0...n keys of the dictionary have no value for i set it to 0
        keys = dict.keys()
        for key in range(keys[0], keys[-1], 1):
            if dict.has_key(key) == False:
                dict[key] = 0

    def dictMedian(self, dict):
        # find the median value of all the values for the keys in dict
        median = float(np.sum(dict.values()))/2
        self.completeDict(dict)
        prev = dict.keys()[0]-1;
        sum = 0
        # find the key or the key-and-a-half that is the median key value
        for key in dict.keys():
            sum = sum + dict[key]
            if sum > median:
                if key > prev:
                    return float(prev)+.5
            elif sum == median:
                return float(key)
            else:
                prev = key

    def writeStats(self, path, g):
        filename = path+'/'+'data_statistics.py'
        counts = self.counts
        try:
            f = open(filename, 'w')
        except IOError:
            print 'cannot open', filename
            
        f.write('stats = {}\n'.format(self.counts))
        
        self.log(g,'\tentails: {}, contradicts: {}, permits: {}\n\n'.format(
        counts['entails'], counts['contradicts'], counts['permits']))
        premMedian = self.dictMedian(self.counts['premLen'])
        hypMedian = self.dictMedian(self.counts['hypLen'])
        exMedian = self.dictMedian(self.counts['exampleLen'])
        self.log(g, '\tMedian length of premise: {:.1f} words\n'
                 .format(premMedian))
        self.log(g, '\tMedian length of hypothesis: {:.1f} words\n'
                 .format(hypMedian))
        self.log(g, '\tMedian length of premise+hypothesis: {:.1f} words\n\n'
        .format(exMedian))
        self.log(g, '\tWrote {}\n\n'.format(filename))
        f.close()
        return
        
    def plotLengths(self, path, type, g):
        filename = path+'/'+type+'_lengths.pdf'
        data = self.counts[type+'Len']
        x = np.array(data.keys())
        y = np.array(data.values())
        self.plot = plt.plot(x, y)
        plt.title('Premise and hypothesis lengths')
        # if type == 'example':
        #     plt.title('# words: premise+hypothesis')
        # elif type == 'prem':
        #     plt.title('# words: premise')
        # else:
        #    plt.title('# words: hypohesis')
        plt.savefig(filename, format='pdf')
        g.write('\tWrote a density plot in {}\n'.format(filename))
        g.write('\tWrote a density plot in {}\n\n'.format(filename))
        return plt
   
    def plotStats(self, path, g):
        prem =self.plotLengths(path, 'prem', g)
        hyp =self.plotLengths(path, 'hyp', g)
        # example = self.plotLengths(path, 'example', g)
        return hyp

        
        
        
