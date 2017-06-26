# -*- coding: utf-8 -*-
from PI import flexibleUtterance
from filedata import dataStore
from PIDict import ImpType
import sys, os, util
from random import random
import codecs
stats = util.stats()
newTypes = {}
count = 0
readdata = False
logpath = '../stats'
readpath = '../sources'
writepath = '../data'
writedata = "-write" in sys.argv
total = 0
header = True
longestType = 0
version = ''

if '-path' in sys.argv:
    i = sys.argv.index('-path')
    readpath = sys.argv[i+1]
    if '_' in readpath:
        pos = readpath.index('_')
        readpath = readpath[:pos]
        version = readpath[pos:]
        writepath = '../data'+version
        logpath = '../stats'+version
        if not os.path.isdir(writepath):
            os.mkdir(writepath, 0755)
    else:
        logpath = '../stats'
        readpath = '../sources'
        writepath = '../data'
        namelist = util.getFiles(readpath)
else:
    namelist = []
    for n in sys.argv[1:]:
        if n[0] != '-':
            namelist.append(n)
            
if '-count' in sys.argv:
    statsdir = '../stats'+version
    if not os.path.isdir(statsdir):
        os.mkdir(statsdir, 0755) 
    g = codecs.open(logpath+'/log.txt', encoding='utf-8', mode='w')

for filename in namelist:
    # print 'Processing {}'.format(filename)
    fd = dataStore()
    fd.stats = stats
    mods = fd.process_source_file(readpath, filename, newTypes)
    #mods = mods.replace(u'~',u'␣')
    impSign = fd.impSign
    if impSign[0] in ['+', '-', 'o']:
        impSign = ' '+impSign
    ind = filename.find('_examples')
    if ind > 0:
        name = filename[:ind] 

    if '-write' in sys.argv:
        count = fd.writeData(writepath, name)
    else:
        count = fd.dumpData(stats)
        
    if '-count' in sys.argv:
        if header:
            if os.environ.has_key('LATEX_DIRECTORY') is False:
                print("Please specify the path to the directory where the LaTex tables are to be written.")
                print("For example: /Users/laurik/Documents/acl-2017-implicatives")
                os.environ['LATEX_DIRECTORY'] = raw_input("LATEX_DIRECTORY: ").strip()     
            latex_path = os.environ['LATEX_DIRECTORY']
            if latex_path[-1] != '/':
                latex_path = latex_path + '/'
            if os.path.isfile(latex_path+"implicative_constructions.tex"):
                os.remove(latex_path+"implicative_constructions.tex")
            f = open(latex_path+"implicative_constructions.tex", 'w')
            stats.log(g, '\n\t\t\t{}\n\n\t{}\t\t{}\t{}\n'
            .format('LOG OF DATA'+version, 'CONSTRUCTION', 'SIGN','COUNT'))
            stats.latex_header(f)
            header = False
        constr = name.replace('_',' ')
        if len(constr) > 15:
            stats.log(g, '\t{:10}\t{}\t{:5}\n'
            .format(constr, impSign, count))
            stats.latex(f, '{} & ${}$ & {}\\\\\n'
            .format(constr, impSign, count))
        else:
            stats.log(g, '\t{:10}\t\t{}\t{:5}\n'
            .format(constr, impSign, count))
            stats.latex(f, '{} & ${}$ & {}\\\\\n'
            .format(constr, impSign, count))
        total = total + count

if '-count' in sys.argv:
    stats.log(g, '\t======================================\n\t{} \
constructions\tTotal:\t{:5} examples\n\n'.format(len(namelist),total))
    stats.writeStats(statsdir, g)
    stats.latex_close(f, len(namelist), total)
    plot = stats.plotStats(statsdir, g)
    g.close()
    f.close()