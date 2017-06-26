import os

class sign:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg
        
    def __repr__(self):
        return 'pos = {}, neg = {}'.format(self.pos, self.neg)
        
    def __str__(self):
        return '{}|{}'.format(self.pos, self.neg)

class constr:
    def __init__(self, name, mods, sign):
        self.name = name
        self.mods = mods
        self.sign = sign
        
    def __repr__(self):
        return 'constr = {}, mods = {}, sign = {}'.format(self.name, self.mods, self.sign)
    
    def __str__(self):
        return '{}, {}, {}'.format(self.name, self.mods, self.sign)

#def get_constr_name(line, file):    
#    if line.find('(') > 0:
#        pos1 = line.index('(')
#        if line.find(')') > 0:
#            pos2 = line.index(')')
#        else:
#            raise IOError("Missing a matching right paren in {}".format(line))
#        pos3 = line.index('[')
#        pref = line[:pos1].strip()
#        suff = line[pos2+1:pos3].strip()
#        name = pref + ' ' + suff 
#        return name
#    else:
#        pos = line.index('[')
#        return line[:pos].strip()

def get_constr_name(file):
    if file.find('_examples.txt') < 0:
        raise IOError("expecting '_examples.txt' in {}".format(file))
    else:
        pos = file.index('_examples.txt')
        name = file[:pos].replace('_', ' ')
    return name

def get_sign(line, file):
    if not line.find(']'):
        raise IOError("No '[x,y]:' signature in {}".format(file))
    pos1 = line.index('[')
    pos2 = line.index(']')
    spl = line[pos1:pos2].strip().strip('[]').split('|')
    return sign(spl[0], spl[1])
    
def get_mods(line, file):
    if line.find('('):
        pos1 = line.index('(')
        if line.find(')'):
            pos2 = line.index(')')
        else:
            raise IOError("No '(x,y,..):' mods in {}".format(file))
        mods =  [x.split()[0] for x in line[pos1+1:pos2].split(',')]
    else:
        mods = []
    
    return mods
    
def get_constr(f, file):
    line = f.readline().strip('\n')
    while line.find('Implicatives:') < 0:
        line = f.readline().strip('\n')
        if line == "":
            raise IOError("No 'Implicative:' line in {}".format(file))
    line = f.readline().strip('\n')

    while not (line.find('[')):
        line = f.readline.strip()
        if line == "":
            raise IOError("No '[x,y]:' signature in {}".format(file))
            
    if line.find('(') > 0:
       mods = get_mods(line, file)
    else:
        mods = []
        
    name = get_constr_name(file)
    sign = get_sign(line, file)
    
    cnstr = constr(name, mods, sign)
    return cnstr
        
def get_constructions(path, dir):
    constrs = []
    files = os.listdir(path+dir)
    for file in files:
        with open(path+dir+file,'r') as f:
            constrs.append(get_constr(f, file))
    return constrs

def sort_constructions(constructions):
    determ = []
    probab = []
    counts =  {"deterministic":0, "probabilistic":0}
    for cnstr in constructions:
        sign = cnstr.sign
        if sign.pos.find('.') < 0 and sign.neg.find('.') < 0:
            determ.append(cnstr.name)
        else:
            probab.append(cnstr.name)
    return determ, probab, counts
       