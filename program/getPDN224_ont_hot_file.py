# encoding: UTF-8
import re
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO
from Bio import SwissProt
import os
from numpy import argmax
from numpy import *



dataset1={}

for seq_record in SeqIO.parse('PDNA-224.fasta', 'fasta'):
    #print('seq_record:  ',seq_record)
    #print(seq_record.seq)
    #print('{} is calculating pssm'.format(seq_record.id))
#     #print('')
#     if os.path.exists(inputfile):
#         os.remove( inputfile)
#     ont_hotfile = "".join( ('One_Hot', '_', seq_record.id, '.txt'))
#     print(ont_hotfile)

#     SeqIO.write( seq_record,inputfile, 'fasta')
    
    data = seq_record.seq
    # define universe of possible input values
    alphabet =  'GAVLIPFYWSTCMNQDEKRH'
    #
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    char_to_int['X'] = 20
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    int_to_char[20] = 'X'
#     print(char_to_int)
#     print(int_to_char)
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
#     #通过正则得出乱码所在位置
#     str1=data
#     word = u'[^GAVLIPFYWSTCMNQDEKRH]'
#     a = [m.start() for m in re.finditer(word, str1)]
#     print(a)

    # one hot 编码
    #print(integer_encoded)
    onehot_encoded = list()
    for value in integer_encoded:
        if value==20:
            letter1=[1/21]*20
            onehot_encoded.append(letter1)
        else:
            letter = [0 for _ in range(20)]
            letter[value] = 1
            onehot_encoded.append(letter)
    onehot_encoded = np.array(onehot_encoded)
    #print(onehot_encoded)
    # invert encoding
    
    dataset1[seq_record.id]=onehot_encoded

    sio.savemat('PDNA-224-onehot.mat',dataset1)


print('over!')