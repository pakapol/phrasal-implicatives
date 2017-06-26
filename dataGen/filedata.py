# -*- coding: utf-8 -*-
import util, re, PIDict, PI
from PI import flexibleUtterance
from random import random
from PIDict import ImpType
import codecs

class dataStore:

    def __init__(self):
        self.exampleList = []
        self.impSign = None
        self.npStore = {'sing':[], 'plur':[]}
        self.vpStore = {'pos':[], 'neg':[]}
        
    def updateImpTypes(self, newTypes):
        change = False
        for key in newTypes.keys():
            newValue = newTypes[key]
            if key in ImpType.keys():
                oldValue = ImpType[key]
                if newValue == oldValue:
                    continue
                else:
                    ImpType[key] = newValue
                    change = True
            else:
                ImpType[key] = newValue
                change = True
        if change:
            try:
                f = codecs.open('PIDict.py', encoding='utf-8', mode='w')
                f.write('ImpType = {}\n'.format(ImpType))
                f.close()
            except  IOError:
                print 'cannot open', filename
                
    def newRule1(self, rule):
        # truncate e.g order(s) to order
        newRule = []
        for item in rule:
            if '(' in item:
                pos = item.index('(')
                newRule.append(item[:pos])
            else:
                newRule.append(item)
        return newRule
        
    def newRule2(self, rule):
        # strip parentheses, order(s) --> orders
        newRule = []
        for item in rule:
            if '(' in item:
                newRule.append(re.sub('[()]','',item))
            else:
                newRule.append(item)
        return newRule
                
    def process_lines(self, f, newTypes):
        mods = ""
        for line in f:
            if '#' in line:
                uncommented = line[:-1][:line.index('#')]
            else:
                uncommented = line[:-1]
            if uncommented == "":
                continue
            if uncommented.lower() == "examples:":
                readdata = True
                continue
            if uncommented.lower() == "implicatives:":
                readdata = False
                continue
            if readdata:
                e = flexibleUtterance(uncommented, ImpType, newTypes)
                self.storeExample(e)
            else:
                rawrules = util.getAllSentences(line[:-1])
                if '(' in line:
                    mods = line[line.index('(')+1:line.index(')')]
                impSign = line.split()[-1][1:-1]
                if self.impSign == None:
                    self.impSign = impSign
                elif self.impSign != impSign:
                    raise Exception('SignError: Conflicting types {} and {} in {}'.
                    format(self.impSign, impSign, name))
                    
                for rule in rawrules:
                    # split e.g. obey order(s) to obey order and obey orders
                    if '(' in rule[-2]:
                        rule1 = self.newRule1(rule)
                        newTypes[tuple(rule1[:-1])] = rule[-1][1:-1]
                        rule2 = self.newRule2(rule)
                        newTypes[tuple(rule2[:-1])] = rule[-1][1:-1]
                    else:
                        newTypes[tuple(rule[:-1])] = rule[-1][1:-1]
                self.updateImpTypes(newTypes)
        return mods

    def process_source_file(self, path, name, newTypes):
        filename = path+'/'+name
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            return self.process_lines(f, newTypes)      
        
    def decodeInfl(self,infl):
        if infl in [u'S1', u'S2', u'S3']:
            return 'sing'
        elif infl in [u'P1', u'P2', u'P3']:
            return 'plur'
        else:
            infl2 = infl.split(',')
            if infl == infl2:
                raise Exception('ParseError: unexpected inflection mark \"{}\"'.format(' '.infl))
            elif infl2[0] in ['F', 'M']:
                if infl2[1] == u'S3':
                    return 'sing'
                elif infl2[1] == 'P3':
                    return 'plur'
                else:
                    raise Exception('ParseError: unexpected inflection mark \"{}\"'.format(' '.infl2))
            else:
                Exception('ParseError: unexpected inflection mark \"{}\"'.format(' '.infl2))

    def storeExample(self, example):
        self.exampleList.append(example)
        for sent in example.sentList:
            infl = sent[0][-1]
            attr = self.decodeInfl(infl)
            if (sent[0] in self.npStore[attr]) == False:
                self.npStore[attr].append(sent[0])
            if u'not' in sent[2]:
                vpst = self.vpStore['neg']
            else:
                vpst = self.vpStore['pos']
            if (sent[2] in vpst) == False:
                vpst.append(sent[2])
        return  
    
    def writeData(self, path, name):
        kind = '_data'
        fsuffs = ['txt']
        filedict= {}
        count = 0
        for suff in fsuffs:
            filename = str(path+'/'+name+kind+'.'+suff)
            try:
                f = codecs.open(filename, encoding='utf-8', mode='w')
                filedict[suff] = f
            except IOError:
                print 'cannot open', filename
        for example in self.exampleList:
            for prem,lab,hyp in example.getAllEntPairs(self):
                filedict['txt'].write(u'{0}\n{1}\n{2}\n\n'.format(prem, lab, hyp))
                count = count+1
        for f in filedict.keys():
            filedict[f].close()
        return count
                

    def dumpData(self, stats):
        count = 0
        for example in self.exampleList:
           for prem,lab,hyp in example.getAllEntPairs(self):
               #wrds = stats.updateStats('premLen', len(prem))
               #wrds = wrds + stats.updateStats('hypLen', len(hyp))
               #stats.updateStats('exampleLen', wrds)
               print u'{0}\n{1}\n{2}\n'.format(prem, lab, hyp)
               count = count+1
        return count

        
                
        
