from PI import flexibleUtterance
from PIDict import VPType
import sys, re, util

addition = {}
while 1:
    inp = str(raw_input(">> "))
    if inp[:4] == "add(" and inp[-1] == ")":
        rawrules = util.getAllSentences(inp[4:-1])
        for r in rawrules:
            addition[tuple(r[:-1])] = r[-1][1:-1]
    elif inp[:5] == "add: ":
        rawrules = util.getAllSentences(inp[5:])
        for r in rawrules:
            addition[tuple(r[:-1])] = r[-1][1:-1]
    else:
        u = flexibleUtterance(inp, VPType, addition)
        for x in u.getAllEntPairs():
            print x
