2# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO
import os

inputfile = 'input.fasta'
for seq_record in SeqIO.parse('PDNA-224.fasta', 'fasta'):
    print('{} is calculating pssm'.format(seq_record.id))
    if os.path.exists(inputfile):
        os.remove( inputfile)
    pssmfile = "".join( ('pssm', '_', seq_record.id, '.txt'))
    SeqIO.write( seq_record, inputfile, 'fasta')
    psiblast_cline = NcbipsiblastCommandline( query = inputfile, db='swissprot', evalue=0.001,
                                             num_iterations=3, out_ascii_pssm=pssmfile)
    stdout,stderr=psiblast_cline()
