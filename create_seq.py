import random
import sys

filename = sys.argv[1]

nt = ['A','C','G','T']

out = ''
for i in range(500):
   out += '>SEQ '+str(i)+'\n'
   for j in range(random.randint(400, 700)):
      for k in range(60):
         out += nt[random.randint(0,3)]
      out += '\n'

with open(filename, 'w+') as f:
   f.write(out)
