import codecs
import sys
readpath = '../sources'
writepath = '../presources'
namelist = []

for n in sys.argv[1:]:
    namelist.append(n)
    
for name in namelist:
    readdir = readpath+'/'+name
    with codecs.open(readpath+'/'+name, encoding='utf-8', mode='r') as f:
        with codecs.open(writepath+'/'+name.replace('_examples.txt', '.txt'),
        encoding='utf-8', mode='w') as g:
            exampleLine = False
            line = f.readline()
            while line:
                words = line.split()
                for wrd in words:
                    if exampleLine and '[' in wrd and not ('Poss' in wrd or 'AND' in wrd):
                        wrd = wrd[0:wrd.index('[')]
                    g.write(wrd)
                    if wrd.lower() == 'examples:':
                        exampleLine = True
                    if wrd != words[-1]:
                        g.write(' ')
                g.write('\n')
                line = f.readline()
            