import argparse
import logging
import re
from os.path import basename

try:
    from modeller import *

    from NumChains import NumChains
except:
    logging.info(f'Align.py: Modeller not installed')


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Modeller at: https://bitbucket.org/lcbio/ca2all/src/master/
# -------------------------------------------------------------------------------------------------------------------------------------


# This demonstrates one way to generate an initial alignment between two (or more) PDB sequences provided as PDB structures.
def Align(file0, file1, file2=None, file3=None):
    ###--- Set Modeller environment (including search patch for model.read())
    env = environ()
    env.io.atom_files_directory = ["."]

    if file2 == None:
        _PIR_TEMPLATE = '\n'.join(
            ['>P1;%s', 'sequence:::::::::', '%s', '*', '']
        )
    else:
        _PIR_TEMPLATE = '\n'.join(
            ['>P1;%s', 'sequence::1:%s:%s:%s::::', '%s', '*', '']
        )

    aa_names = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    aa_names = {v: k for k, v in aa_names.items()}
    pattern = re.compile('ATOM.{9}CA .([A-Z]{3}) ([A-Z ])(.{5}).{27}(.{12}).*')
    prefix = basename(file1).rsplit('.', 1)[0]
    seq = ''
    atoms = []  # structure of atoms is: ('GLU', 'A', ' 207 ', '  1.00 37.96')
    chains = []

    ####--- PIR FILE PREPARATION---####
    with open(file0, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            else:
                match = re.match(pattern, line)
                if match:
                    atoms.append(match.groups())
    if not len(atoms):
        raise Exception('File %s contains no CA atoms' % file1)
    set = NumChains(file1)
    rr = int(atoms[0][2]) - 1
    for a in atoms:
        s, c, r = a[:3]
        if not len(chains):
            chains += c
        if c not in chains:
            chains += c
            seq += '/'
        elif int(r) != int(rr) + 1:
            seq += '/'
        rr = r
        seq += aa_names[s]

    with open(file3, 'w') as ff:
        if file2 == None:
            ff.write(_PIR_TEMPLATE % (prefix, seq))
        else:
            ff.write(_PIR_TEMPLATE % (prefix, set[1], set[2], set[3], seq))

    # Create a new empty alignment and model:
    aln = alignment(env)
    mdl = model(env)

    # Read the whole CA-trace file and add the model sequence to alignment
    mdl.read(file=file1, model_segment=('FIRST:@', 'END:'))
    aln.append_model(mdl, align_codes='initial', atom_files=file1)

    # Read template (all-atom) file and add to alignment
    if file2 != None:
        mdl.read(file=file2, model_segment=('FIRST:@', 'END:'))
        aln.append_model(mdl, align_codes='template', atom_files=file2)

    # Align them by sequence
    aln.malign(gap_penalties_1d=(-500, -300))
    pir = open(file3, 'a+')
    aln.write(file=pir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Align',
        description="""
Program creates sequence alignment based on provided structures.
The target sequence comes from the first PDB file. Note that the 
residues indexes (ids) decide about chain breaks. At least one more PDB 
file (-f1 option) is required to prepare alignment. """,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=''
    )
    parser.add_argument(
        '-s', '--file0-pdb',
        help='input pdb file: sequence with proper breaks based on residues ids',
        metavar='SEQ',
        dest='f_0',
        required=True
    )
    parser.add_argument(
        '-f1', '--file1-pdb',
        help='input pdb file: CA-trace structure',
        metavar='FILE1',
        dest='f_1',
        required=True
    )
    parser.add_argument(
        '-f2', '--file2-pdb',
        help='input pdb file: template structure',
        metavar='FILE2',
        dest='f_2',
        default=None
    )
    parser.add_argument(
        '-f3', '--file3-ali',
        help='output file: alignment',
        metavar='FILE3',
        dest='f_3',
        required=True
    )
    args = parser.parse_args()
    Align(args.f_0, args.f_1, args.f_2, args.f_3)
