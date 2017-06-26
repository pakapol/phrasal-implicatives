# -*- coding: utf-8 -*-
import util, sys, re, PIDict
from copy import deepcopy
from random import random, randint

class flexibleUtterance:

    def __init__(self, inputstring, *dicts):
        #list of (NP,IMP,VP, dicts)
        self.stats = False # to be set from the calling makefile.py file
        self.sentList = [util.simpleParse(s) for s in util.getAllSentences(inputstring)]
        self.constr, self.entailmentProb = util.lookupPI(self.sentList[0][1], *dicts)
        
    def __repr__(self):
        return 'stats = {}, sentlist = {}, constr = {}, entailmentProb = {}'.format(self.stats,
        self.sentList, self.constr, self.entailmentProb)
        
    def __str__(self):
        return '{}, {}, {}, {}'.format(self.stats, self.sentList, self.constr, self.entailmentProb)


    def getPremise(self, tense, neg=False):
        result = []
        for NPtoken,IMPtoken,VPtoken in self.sentList:
            premise, negForm = util.toPremise(NP=NPtoken[:-1],IMP=IMPtoken,VP=VPtoken,tense=tense,PERSON=NPtoken[-1],neg=neg)
            result.append(premise)
            
        return result, negForm

    def getHypothesis(self, tense, negPremise=False, contradiction=False, negForm=None):
        result = []
        hypneg = (self.entailmentProb[int(negPremise)] < 0.0)
        for NPtoken,_,VPtoken in self.sentList:
            hypothesis = util.toHypothesis(NP=NPtoken[:-1],VP=VPtoken,tense=tense,PERSON=NPtoken[-1],neg=hypneg^contradiction,
            negForm=negForm)
            result.append(hypothesis)
        return result
        
    def npNumber(self, np):
        if np[-1] in [u'S1', u'S2', u'S3', u'F,S3', u'M,S3']:
            return 'sing'
        else:
            return 'plur'
    
    def differentNPs(self, np, newNP):
        if len(np) != len(newNP) or np[:-1] != newNP[:-1]:
            return True
        else:
            return False
    
    def isPronoun(self, np, otherType):
        # don't allow the substitution of he, she, it or they for NPs that
        # they could refer to. First and second person pronoun substitutions are
        # always OK
        if np == u'I' or np == u'you' or np == u'we':
            return True
        elif (np == u'he' and otherType == 'M,S3') or (np == 'she' and otherType == 'F,S3'):
            return False
        elif np == u'it' and otherType == 'S3' or np == 'they' and otherType == 'P3':
            return False
        else:
            return False
            
    def subjPro_to_objPro(self, pro):
        if pro == u'I':
            return u'me'
        elif pro == u'she':
            return u'her'
        elif pro == u'he':
            return u'him'
        elif pro == 'we':
            return u'us'
        elif pro == u'they':
            return u'them'
        else:
            return pro
        
    def bothSingular(self, oldType, newType):
        if ('S3' in oldType and 'S3' in newType):
            return True
        else:
            return False
    
    def compatibleNPs(self, np, newNP):
        # Avoid substituting 'it' and 'they' for NPs they could refer to
        # Let other pronouns substitute for any NP, and vice versa
        oldType = np[-1]
        newType = newNP[-1]
        if ('F' in oldType and 'M' in newType) or ('M' in oldType and 'F' in newType):
            return True
        elif (self.isPronoun(np[0], newType) or self.isPronoun(newNP[0], oldType)) and oldType == newType:
            return True
        elif (newNP[0] == u'he' and oldType == 'M,S3') or (newNP[0] == u'she' and oldType == 'F,S3'):
            return False
        elif (newNP[0] == u'it' and oldType == 'S3') or (newNP[0] == u'they' and oldType == 'P3'):
            return False   
        elif oldType == newType and self.differentNPs(np, newNP):
            return True
        #elif np != u'it' and newNP != u'it' and self.bothSingular(oldType, newType):
         #   return True   
        else:
            return True
    
    def overlap(self, np, newNP):
        np_words = np[:-1]
        newNP_words = newNP[:-1]
        for wrd in np_words:
            if wrd in newNP_words:
                return True
        return False
        
        
    def someoneElse(self, code):
        # different NP of last resourt
        if code == u'MS3' or code == u'F,S3':
            return ['someone else', code]
        elif code == u'S3':
            return ['something else', code]
        elif code == u'P3':
            return ['some others', code]
        elif code == 'S2' or code == 'P2':
            return ['someone other than you', code]
        else:
            return ['someone other than me', code]
    
    def anotherNP(self, np, npStore):
        choices = npStore[self.npNumber(np)]
        newNP = np
        tries = 0
        limit = len(choices)- 1
        max_tries = 2 * limit
        while ((self.overlap(np, newNP) or not self.compatibleNPs(np, newNP)) and tries < max_tries):
            newNP = choices[randint(0, limit)]
            tries = tries+1
        if newNP == np:
            newNP = self.someoneElse(np[:-1])
        return newNP
            
    def doSomethingElse(self):
        # different VP of last resort
        return [u'to', '$V', u'do', u'something', u'completely', u'different']   
        
    def replaceBe(self, verb, ind, do):
        verb[ind] = do
        verb[-1] = util.bareform(verb[-1])
        return verb
        
    def activizeNegPast(self, verb, negIndex):
        aux = verb[negIndex-1]
        if aux == 'was':
            return self.replaceBe(verb, negIndex-1, "did")
        elif aux == 'were':
            return self.replaceBe(verb, negIndex-1, "did")
        elif aux == 'am':
            return self.replaceBe(verb, negIndex-1, "do")
        elif aux == 'are':
            return self.replaceBe(verb, negIndex-1, "do")
        elif aux == 'is':
            return self.replaceBe(verb, negIndex-1, "does")
        else:
            return verb
            
    def activizePosPast(self, verb):
        # eliminates the form of "be" before the passive form
        for wrd in verb[:-1]:
            if wrd == u'were' or wrd == u'was':
                verb.remove(wrd)
        return verb 
        
    def depassivize(self, subj, prem, tense, negForm, ctxt):
        # Get a new subject, find the initial verbal span up to the passive verb,
        # append the old subject tranformed to an object and concatenate the
        # new verbal phrase with the original vp.

        oldSubj = subj[:-1]
        # take into account the number and gender of the original subject
        newSubj = self.anotherNP(subj, ctxt.npStore)
        result = newSubj[:-1]
        
        if oldSubj in [[u'I'], [u'i'], [u'he'], [u'she'], [u'we'], [u'they']]:
            obj = [self.subjPro_to_objPro(subj[0])]
        else:
            obj = oldSubj
        
        tail = prem[len(oldSubj):]
        verb = tail[:tail.index(self.constr[-1])+1]
        vp = tail[len(verb):]
        
        # In the simple past tense, if the passive contains negation,
        # replace the passive participle by its infinitive form and
        # replace the form of "be" in front of the negations with
        # the corresponding form of "do"
        # e.g. "was not forced" --> "did not force"
        
        if negForm == 'not':
            negIndex = verb.index('not')
        elif negForm == "n't":
            negIndex = verb.index("n't")
        else:
            negIndex = -1
        
        if tense == 'Past.Simple':
            if negIndex > 0:
                self.activizeNegPast(verb, negIndex)
            else:
                self.activizePosPast(verb)
        else:        
            # remove the auxiliary be
            verb.remove(u'been')
        
        result = result + verb + obj + vp
    
        return(result)
        
        
    def anotherVP(self, vp, vpStore):
        newVP = vp
        tries = 0
        if u'not' in vp:
            vpst = vpStore['neg']
        else:
            vpst = vpStore['pos']
        limit = len(vpst)
        max_tries = 2 * limit
        while newVP == vp and tries < max_tries:
            newVP = vpst[randint(0, len(vpst)-1)]
            tries = tries+1
        if newVP == vp:
            newVP = self.doSomethingElse()
        return newVP
                
    def permitOtherSubject(self, ctxt, prem, hyp, tense, negPremise, contradiction, negForm):
            result = []
            hyp = []
            hypneg = (self.entailmentProb[int(negPremise)] < 0.0)
            for NPtoken, _, VPtoken in self.sentList:
                newNP = self.anotherNP(NPtoken, ctxt.npStore)
                hyp.append(util.toHypothesis(NP=newNP[:-1],VP=VPtoken,tense=tense,PERSON=newNP[-1],neg=hypneg^contradiction,
                negForm=negForm))
            for i in xrange(len(hyp)):
                ctxt.stats.updateStats('exampleLen', len(prem)+len(hyp))
                # sys.stdout.write("{}\n {}\n\n".format(" ".join(prem), " ".join(hyp[i])))
                result.append((" ".join(prem), 'permits', " ".join(hyp[i])))       
            return result

    def permitOtherVP(self, ctxt, prem, hyp, tense, negPremise, contradiction, negForm):
            result = []
            hyp = []
            hypneg = (self.entailmentProb[int(negPremise)] < 0.0)
            for NPtoken, _, VPtoken in self.sentList:
                newVP = self.anotherVP(VPtoken, ctxt.vpStore)
                hyp.append(util.toHypothesis(NP=NPtoken[:-1],VP=newVP,tense=tense,PERSON=NPtoken[-1],neg=hypneg^contradiction,
                negForm=negForm))
            for i in xrange(len(hyp)):
                ctxt.stats.updateStats('exampleLen', len(prem)+len(hyp))
                result.append((" ".join(prem), 'permits', " ".join(hyp[i])))      
            return result

    def permits(self, ctxt, prem, hyp, tense, negPremise, contradiction, negForm):
        #return self.permitOtherSubject(ctxt, prem, hyp, tense, negPremise, contradiction)
        if random() < .5:
            return self.permitOtherSubject(ctxt, prem, hyp, tense, negPremise, contradiction, negForm)
        else:
            return self.permitOtherVP(ctxt, prem, hyp, tense, negPremise, contradiction, negForm)
            
    def make_example(self, prem, rel, hyp, stats):
        example = ((" ".join(prem), rel, " ".join(hyp)))
        pLen = len(prem)
        hLen = len(hyp)
        stats.addToRelation(rel)
        stats.updateStats('premLen', pLen)
        stats.updateStats('hypLen', hLen)
        stats.updateStats('exampleLen', pLen + hLen)
        stats.updateStats('prem-hypLen', pLen - hLen)
        return example
              
    def getAllEntPairs(self, ctxt):
        result = []
        stats = ctxt.stats
        for negPremise in [False,True]:
            # if the construction is a two-way implicative, [+\-] or [-|+], recoded as [1.0|-1.0] probabilities,
            # we don't change anything. The same goes for one-way-implicatives coded as [+|o], [-|o].
            # If the construction is a biased one-way implicative, say [+\.f], [-\.f], [.f|-], or [.f|+],
            # we calculate a floating point random number .r. If .r < .f, we interpret .f as o (= 'permits'),
            # otherwise we interpret .f as - (= 'contradicts, -1.0) in a negPremise, that is, in [+/.f] and [-/.f],
            # and in the positive case as in  [.f/+] and [.f/-]. The interpretation of .f is constant for all the tenses of the same example. It may vary for
            # other examples that are variants of the same original example.
            # For instance, with the signature [.4/-] for
            # the construciont 'have chance', examples like
            #     I did not have a chance to win the lottery
            #     I did not have my chance to win the lottery
            # agree that in in the negative, I did not have a/my chance to win the lottery
            # they both entail that I did not win the lottery.
            # However, in the case of positive polarity, the two examples might turn out differently. That is,
            #     I had a chance to win the lottery
            #     I had my chance to win the lottery
            # one randomly selecting 'entails' (= 1.0) and the other selecting o ('permits').
            # Whichever choice is made for a particular example, it is made for all tense forms of the same example.
            # If the relation between the premise and hypothesis is 'entails', the hypothesis permits the premise.
            # If the relation between the premise and the hypothesis is 'contradicts' then the hypothesis also contradicts
            # the premise.            
            choice = self.entailmentProb[int(negPremise)] 
            rand = random()
            if choice not in [u'o', 1.0, -1.0]:
                if rand < abs(choice):
                    if negPremise:
                        choice = -1.0
                    else:
                        choice = 1.0
                else:
                    choice = u'o'

            for contradiction in [False,True]:
                for tense in [['Past.Simple', 'Present.Perfect','Past.Perfect'][randint(0,2)]]:
                    prem, negForm = self.getPremise(tense,neg=negPremise)
                    hyp = self.getHypothesis(tense,negPremise=negPremise,contradiction=contradiction, negForm=negForm)
                    if choice == u'o':
                        con = 'permits'
                    elif contradiction:
                        con = 'contradicts'
                    else:
                        con = 'entails'
                    # If the construction is one of the passive types, construct a corresponding active
                    # premise. For example 'we were forced to leave' ==> 'Mary forced us to leave.'
                    # The hypothesis stays the same.
                    if self.constr in [(u'be', u'forced'), (u'be',u'prevented')]:
                        # Need to match the number and conjugation that goes with the old subject.
                        oldSubj = self.sentList[0][0]
                        prem.append(self.depassivize(oldSubj, prem[0], tense, negForm, ctxt))
                        hyp.append(hyp[0])
                    for i in xrange(len(prem)):
                        result.append(self.make_example(prem[i], con, hyp[i], stats))
                        if contradiction:
                            result.append(self.make_example(hyp[i], con, prem[i], stats))
                        elif con == 'entails':
                            result.append(self.make_example(hyp[i], 'permits', prem[i], stats))
                        # for 'entails' and 'contradicts' generate a 'permit' half of the time
                        if choice in [1.0, -1.0] and randint(0,1) > 0:
                            permits = self.permits(ctxt, prem[i], hyp[i], tense, negPremise, contradiction, negForm)
                            for j in xrange(len(permits)):
                                result.append(permits[j])
                                stats.addToRelation('permits') 
                                           
        return result
