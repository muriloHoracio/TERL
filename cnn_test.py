#! /usr/bin/env python
import sys
from test_parser import get_options
from test_parser import print_options

options = get_options(sys.argv[1:])
print_options(options)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

LONGER_SEQ_ERROR = '\nERROR:\n\nFile {fl} has a sequence with length longer ({longer}) then the max_len ({max_len}) permited by the model'

int_to_nucleotide = {
	1: 'A',
	2: 'C',
	3: 'G',
	4: 'T',
	5: 'N',
	0: ''
}

nucleotide_to_int = {
	'A': 1,
	'C': 2,
	'G': 3,
	'T': 4,
	'N': 5
}

classification = {
	'Copia': ['Class I\tLTR\tCopia',0],
	'Gypsy': ['Class I\tLTR\tGypsy',0],
	'Bel-Pao': ['Class I\tLTR\tBel-Pao',0],
	'Retrovirus': ['Class I\tLTR\tRetrovirus',0],
	'ERV': ['Class I\tLTR\tERV',0],
	'Dirs': ['Class I\tDIRS\tDirs',0],
    'Ngaro': ['Class I\tDIRS\tNgaro',0],
	'VIPER': ['Class I\tDIRS\tVIPER',0],
	'Penelope': ['Class I\tPLE\tPenelope',0],
	'R2': ['Class I\tLINE\tR2',0],
	'RTE': ['Class I\tLINE\tRTE',0],
	'Jockey': ['Class I\tLINE\tJockey',0],
	'L1': ['Class I\tLINE\tL1',0],
	'I': ['Class I\tLINE\tI',0],
	'tRNA': ['Class I\tSINE\ttRNA',0],
	'7SL': ['Class I\tSINE\t7SL',0],
	'5S': ['Class I\tSINE\t5S',0],
	'Mariner': ['Class II\tSubclass 1\tTIR\tTc1-Mariner',0],
	'hAT': ['Class II\tSubclass 1\tTIR\thAT',0],
	'Mutator': ['Class II\tSubclass 1\tTIR\tMutator',0],
	'Merlin': ['Class II\tSubclass 1\tTIR\tMerlin',0],
	'Transib': ['Class II\tSubclass 1\tTIR\tTransib',0],
	'P': ['Class II\tSubclass 1\tTIR\tP',0],
	'PiggyBac': ['Class II\tSubclass 1\tTIR\tPiggyBac',0],
	'PIF-Harbinger': ['Class II\tSubclass 1\tTIR\tPIF-Harbinger',0],
	'CACTA': ['Class II\tSubclass 1\tTIR\tCACTA',0],
	'Crypton': ['Class II\tSubclass 1\tCrypton\tCrypton',0],
	'Helitron': ['Class II\tSubclass 2\tHelitron\tHelitron',0],
	'Maverick': ['Class II\tSubclass 2\tMaverick\tMaverick',0],
	'LTR': ['Class I\tLTR',0],
	'DIRS': ['Class I\tDIRS',0],
	'PLE': ['Class I\tPLE',0],
	'LINE': ['Class I\tLINE',0],
	'SINE': ['Class I\tSINE',0],
	'TIR': ['Class II\tSubclass 1\tTIR',0],
	'Crypton': ['Class II\tSubclass 1\tCrypton',0],
	'Helitron': ['Class II\tSubclass 2\tHelitron',0],
	'Maverick': ['Class II\tSubclass 2\tMaverick',0],
	'Subclass 1': ['Class II\tSubclass 1',0],
	'Subclass 2': ['Class II\tSubclass 2',0],
	'Class I': ['Class I',0],
	'Class II': ['Class II',0]
}

def print_model(architecture, functions, widths, strides, feature_maps, max_len):
	print('*' * 79 + '\n**' + ' ' * 34 + ' MODEL ' + ' ' * 34 + '**\n' + '*' * 79)
	print('%20s %s' % ('Classes:',', '.join(t for t in classes)))
	print('%20s %d' % ('Max length:',max_len))
	print('%20s %s' % ('Architecture:',''.join('%-8s' % t for t in architecture)))
	print('%20s %s' % ('Functions:',''.join('%-8s' % t for t in functions)))
	print('%20s %s' % ('Widths:',''.join('%-8s' % t for t in widths)))
	print('%20s %s' % ('Strides:',''.join('%-8s' % t for t in strides)))
	feature_maps_string = '%20s ' % 'Feature maps:'
	j = 0
	for i, layer in enumerate(architecture):
		if layer == 'conv':
			feature_maps_string += '%-8s' % str(feature_maps[j])
			j += 1
		else:
			feature_maps_string += '%-8s' % '-'
	print(feature_maps_string)

def get_data(files, max_len):
	seqs = []
	headers = []
	for fl in files:
		with open(fl,'r') as f:
			seq = ''
			for l in f.readlines():
				if l[0] == '>':
					headers.append(l)
					if seq != '':
						seqs.append(np.array([nucleotide_to_int[c] for c in seq], dtype=np.uint8))
						if len(seqs[-1]) <= max_len:
							seqs[-1] = np.pad(seqs[-1], (0, max_len - len(seqs[-1])), 'constant', constant_values=(0, 0))
						else:
							print(LONGER_SEQ_ERROR.format(fl=fl, longer=len(seqs[-1]), max_len=max_len))
							seqs[-1] = seqs[-1][0:max_len]
#							exit(-1)
					seq = ''
				else:
					seq += l.upper().strip()
			seqs.append(np.array([nucleotide_to_int[c] for c in seq], dtype=np.uint8))
			if len(seqs[-1]) <= max_len:
				seqs[-1] = np.pad(seqs[-1], (0, max_len - len(seqs[-1])), 'constant', constant_values=(0, 0))
			else:
				print(LONGER_SEQ_ERROR.format(fl=fl, longer=len(seqs[-1]), max_len=max_len))
				seqs[-1] = seqs[-1][0:max_len]
#				exit(-1)
	return np.array([e for e in seqs])

with tf.Session(graph=tf.Graph()) as sess:
	tf.saved_model.loader.load(
		sess,
		['serve'],
		options.model
	)

	num_classes = sess.run('num_classes:0')
	classes = [c.decode('utf-8') for c in sess.run('classes:0')]
	architecture = [layer.decode('utf-8') for layer in sess.run('architecture:0')]
	functions = [func.decode('utf-8') for func in sess.run('functions:0')]
	widths = sess.run('widths:0')
	strides = sess.run('strides:0')
	feature_maps = sess.run('feature_maps:0')
	vocab_size = sess.run('vocab_size:0')
	max_len = sess.run('max_len:0')
	print_model(architecture, functions, widths, strides, feature_maps, max_len)

	x = get_data(options.files, max_len)
	test_size = len(x)

	predictions = np.array([], dtype=np.uint8)
	for batch in range(0, test_size, options.batch_size):
		x_batch = x[batch : batch + options.batch_size]

		pre_xo = sess.run('one_hot_x:0', feed_dict={'pre_x:0': x_batch})
		x_batch = pre_xo.reshape(x_batch.shape[0], max_len, vocab_size, 1)

		predictions = np.concatenate([predictions, sess.run('prediction:0', feed_dict={'x_input:0': x_batch})])
		
	out = ''
	x = list(x)
	for i, pred in enumerate(predictions):
		classification[classes[pred]][1] += 1
		out += '>' + classification[classes[pred]][0] + '\t' + str(classification[classes[pred]][1]) + '\n'
		x[i] = ''.join([int_to_nucleotide[e] for e in x[i]])
		for chunk in range(0, len(x[i]), 60):
			out += x[i][chunk: chunk + 60] + '\n'
	with open('classification_file.fa','w+') as f:
		f.write(out)
