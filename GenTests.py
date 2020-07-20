#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('size', type=int)

args = parser.parse_args()

n = args.size

f = open("Test{0:03d}.pl".format(n), 'w')

inX = ["in{0:03d}".format(i) for i in range(1,n+1)]
onX = ["on{0:03d}".format(i) for i in range(1,n+1)]

# Fluents
f.write("fluents(")
f.write(str(inX + onX).replace(' ','').replace('\'',''))
f.write(").\n")

# Dynamic Laws
for i in range(1,n):
    f.write("dynamicLaw(move,[in{1:03d}],[in{0:03d}]).\n".format(i,i+1))
    
# Sensing Actions
for i in range(1,n+1):
    f.write("senseAct(sense{0:03d},on{0:03d}).\n".format(i))

# Executability Conditions
f.write("exeCnd(move,[]).\n")
for i in range(1,n+1):
    f.write("exeCnd(sense{0:03d},[in{0:03d}]).\n".format(i))
    
# Static Causal Laws
for i in range(1,n+1):
    for j in list(range(1,i)) + list(range(i+1,n+1)) :
        f.write("staticLaw([-in{0:03d}],[in{1:03d}]).\n".format(i,j))

# History 
sms = ["(sense{0:03d},[[on{0:03d}]]),(move,[])".format(i) for i in range(1,n)]
sms = sms + ["(sense{0:03d},[[-on{0:03d}]])".format(i)]
f.write("history(" + str(sms).replace(' ','').replace('\'','') + ").\n")

# Initially
i = ["in{0:03d}".format(1)] + onX
f.write("initially(["+ str(i).replace(' ','').replace('\'','') + "]).\n")
    
