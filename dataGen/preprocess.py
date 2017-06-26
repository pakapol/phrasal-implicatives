# -*- coding: utf-8 -*-
import sys
from util import getFiles, getAllSentences, backToString
import pattern.en as clips
import codecs
from namedict import nameRegistry

subjProCodes = {'I':'I[S1]', 'you':'you[S2]', 'he':'he[M,S3]', 'she':'she[F,S3]', 'it':'it[S3]',
'we':'we[P1]', 'they':'they[P3]', 'people':'people[P3]'}

subjPros = subjProCodes.keys()
impCode = {'before':[], 'imp':[], 'after':[]}

namelist =[]

sourcedir = '../presources'
writedir = '../sources'

def next_line(f, g, i):
    try:
        line = f.next()
        i = i + 1
        return line, i
    except:
        return 'EOF', i
    
def next_example(f, g, i):
    line, i = next_line(f, g, i)
    while line.strip() == '':
        line, i = next_line(f,g, i)
    return line, i
        
def findCode(line):
    before = []
    after = []
    splitLine = line.split()
    for i in range(0, len(splitLine)):
        item = splitLine[i]
        if item[0] == '(' and item[-1] == ')':
            imp = item[1:-1].split(',')
            after = splitLine[i+1:-1]
            break
        elif item[0] == '[':
            imp = item
            break
        else:
            before.append(item)
    impCode['before'] = before
    impCode['imp'] = imp
    impCode['after'] = after
            
def get_implicatives(f, g, i):
    line,i  = next_line(f, g, i)
    while line.strip().lower() != "implicatives:":
        line,i  = next_line(f,g, i)
    g.write(line)
    line,i  = next_line(f, g, i)
    g.write(line)
    findCode(line)
    return line, i
    
def get_examples(f, g, i):
    line,i  = next_line(f, g,i)
    while line.strip().lower() != "examples:":
        g.write(line)
        line,i  = next_line(f, g, i)
    g.write(line)
    return line, i
    
def get_verb(f, g, i):
    line, i = get_implicatives(f, g, i)
    construction = getAllSentences(line[:-1])
    wrd =  construction[0][0]
    verb = clips.conjugate(wrd, tense='past')
    return verb, i
    
def mark_verb(verb, words, ex, i):
    if verb in words:
        pos = words.index(verb)
        words[pos] = verb+'[V]'
    elif verb == u'was':
        verb = 'were'
        if verb in words:
            pos = words.index(verb)
            words[pos] = verb+'[V]'
    else:
        raise Exception("TypoError: No '{}' on line {}: {}".format(verb, i, ex))
        
    return words, pos
    
def help_subj():
    print 'Please type m for [M,S3], f for [F,S3], s for [S3] and p for [P3]'
    print '  or h to see the example'
   
def askUser(words, vpos, ex):
    span = words[:vpos]
    key = tuple(span)
    pos = words.index(span[-1])
    if key in nameRegistry['data'].keys():
        mark =nameRegistry['data'][key]
    else:
        if nameRegistry['modified'] == False:
            print 'Asking for codes for non-pronominal subject NPs'
            help_subj() 
        code = raw_input('{}\nCode: '.format(span)).lower()
        while code not in ['f', 'm', 's', 'p']:
            if code == 'h':
                print ex
                help_subj()
                code = raw_input('{}\nCode: '.format(span)).lower()                
        if code == 'm' or code.lower() == 'f':
            code = code.upper()+',S3'
        elif code == 'p' or code.lower() == 's':
            code = code.upper()+'3'
        mark = '['+code+']'
        nameRegistry['data'][key] = mark 
        nameRegistry['modified'] = True
    words[pos] = words[pos]+mark
    return mark, words
      
def countPros(span):
    pros = 0
    for wrd in span:
        if wrd in subjProCodes:
            pros = pros+1
    return pros
            
      
def mark_subj(words, vpos, ex):
    Found = False
    span = words[:vpos]
    for i in range(0,len(span)):
        if '(' in span[i]:
            span = span[:i]
            vpos = i
            break;
    pros = countPros(span)
    if 'and' in span:
        if 'I' in span or 'me' in span or 'myself' in span:
            words[vpos-1] = words[vpos-1]+'[P1]'
            nameRegistry[tuple(span)] = '[P1]'
            return '[P1]', words
        else:
            return askUser(words, vpos, ex)
    elif 'or' in span:
        return askUser(words, vpos, ex)
    elif 'we,' in span or 'we' in span:
        words[vpos-1] = words[vpos-1]+'[P1]'
        nameRegistry[tuple(span)] = '[P1]'
        return '[P1]', words
    elif len(span) == 1 or (span[0] in subjProCodes and pros == 1):
        # avoid marking Charles I as [S1] but mark I in I (even) as [S1]
        if span[0] in subjPros:
            pos = words.index(span[0])
            mark = subjProCodes[span[0]]
            words[pos] = mark
            mark = mark[mark.index('['):] #strip the pronoun itself
            Found = True
    if Found:
        return mark, words
    else:
        return askUser(words, vpos, ex)

def mark_VP_verb(words, vpos):
    if u'to' in words[vpos+1:]:
        for wrd in words[words.index(u'to')+1:]:
            if wrd == 'vowed':
                print wrd 
            # Workaround for a bug in clips.verbs. It does not have
            # 'use' listed as a verb but can inflect it as regular verb
            # when called to do so
            if wrd in clips.verbs or wrd == u'use':
                pos = words.index(wrd)
                words[pos] = words[pos]+'[V]'
                break
    return words
    
def spliceInPossPro(vpos, words):
    # if the word following the verb is 'a' or 'the', replace it with [PossPro]
    if words[vpos+1] in impCode['imp']:
        vp = words[vpos+1] = '[PossPro]'       
    return words
    
def process_examples(verb, f, g, i):
    ex, i = next_example(f, g, i)
    while ex != 'EOF':
        words = ex.split()
        words, vpos = mark_verb(verb, words, ex, i)
        subjCode, words = mark_subj(words, vpos, ex)
        words = mark_VP_verb(words, vpos)
        g.write(backToString(words))
        g.write('\n')   
        
        if 'PossPro' in impCode['imp']:
            words = spliceInPossPro(vpos, words)
            g.write(backToString(words))
            g.write('\n')

        ex, i = next_example(f, g, i)

def process_file(f, g):
    verb, i = get_verb(f, g, 0)
    line, i = get_examples(f, g, 0)
    process_examples(verb,f, g, i)     

if '-path' in sys.argv:
    i = sys.argv.index('-path') 
    namelist = getFiles(sys.argv[i+1])
else:
    namelist = []
    for n in sys.argv[1:]:
        if n[0] != '-':
            namelist.append(n)
            

for name in namelist:
    print 'Processing {}'.format(name)
    with codecs.open(sourcedir+'/'+name, encoding='utf-8', mode='r') as f:
        if '-write' in sys.argv[1:]:
            with codecs.open(writedir+'/'+name.replace('.txt','_examples.txt'),
            encoding='utf-8', mode='w') as g:
                process_file(f, g)                 
        else:
            process_file(f, sys.stdout)
# if we have acquired new subject codes for nonpronominal NPs save the for future use
if nameRegistry['modified'] == True:
    nameRegistry['modified'] = False
    with codecs.open('namedict.py', encoding='utf-8', mode='w') as f:
        f.write('nameRegistry = {}\n'.format(nameRegistry))

