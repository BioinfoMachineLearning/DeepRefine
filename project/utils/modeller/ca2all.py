import glob
import logging
import os
import re
import sys
from os.path import basename
from tempfile import mkstemp

try:
    from modeller import *
    from modeller.automodel import *
except:
    logging.info(f'ca2all.py: Modeller not installed')

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Modeller at: https://bitbucket.org/lcbio/ca2all/src/master/
# -------------------------------------------------------------------------------------------------------------------------------------

_PIR_TEMPLATE = '\n'.join(
    ['>P1;%s', 'sequence:::::::::', '%s', '*', '', '>P1;model_ca', 'structure:%s:FIRST:@:END:@::::', '*']
)


def ca2all(filename, output=None, iterations=1, verbose=False):
    old_stdout = sys.stdout
    if verbose:
        sys.stdout = sys.stderr
    else:
        sys.stdout = open(os.devnull, 'w')

    pdb = mkstemp(prefix='.', suffix='.pdb', dir='.', text=True)[1]
    prefix = basename(pdb).rsplit('.', 1)[0]
    aa_names = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    aa_names = {v: k for k, v in aa_names.items()}
    atoms = []
    pattern = re.compile('ATOM.{9}CA .([A-Z]{3}) ([A-Z ])(.{5}).{27}(.{12}).*')

    try:
        with open(filename, 'r') as f, open(pdb, 'w') as tmp:
            for line in f:
                if line.startswith('ENDMDL'):
                    break
                else:
                    match = re.match(pattern, line)
                    if match:
                        atoms.append(match.groups())
                        tmp.write(line)
        if not len(atoms):
            raise Exception('File %s contains no CA atoms' % filename)
        chains = [atoms[0][1]]
        seq = ''
        rr = int(atoms[0][2]) - 1
        for a in atoms:
            s, c, r = a[:3]
            if int(r) != int(rr) + 1:
                seq += '/'
            rr = r
            seq += aa_names[s]
            if c not in chains:
                chains += c
        pir = prefix + '.pir'
        with open(pir, 'w') as f:
            f.write(_PIR_TEMPLATE % (prefix, seq, pdb))
        #####---MODELLER SECTION---#####
        env = environ()
        env.io.atom_files_directory = ['.']
        env.libs.topology.read(file='$(LIB)/top_allh.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')

        class MyModel(automodel):
            def special_patches(self, aln):
                self.rename_segments(segment_ids=chains)

        mdl = MyModel(
            env,
            alnfile=pir,
            knowns='model_ca',
            sequence=prefix,
            assess_methods=assess.DOPE
        )
        mdl.md_level = refine.very_fast
        mdl.auto_align(matrix_file=prefix + '.mat')
        mdl.starting_model = 1
        mdl.ending_model = int(iterations)
        mdl.final_malign3d = False
        mdl.make()
        models = [m for m in mdl.outputs if m['failure'] is None]
        cmp_key = 'DOPE score'
        models.sort(key=lambda x: x[cmp_key])
        final = models[0]['name'].rsplit('.', 1)[0] + '.pdb'  # Note: Append _fit.pdb instead when using refine.slow
        sys.stdout = old_stdout

        if output:
            outfile = open(output, 'w')
        else:
            outfile = sys.stdout
        with open(final) as f:
            a = iter(atoms)
            current = ch = r = t = nl = None
            for line in f:
                if line.startswith('ATOM'):
                    res = line[21:27]
                    if not current or current != res:
                        current = res
                        ch, r, t = a.__next__()[1:]
                    nl = line[:21] + ch + r + line[27:54] + t
                    if len(line) > 66:
                        nl += line[66:]
                    outfile.write(nl)
                elif line.startswith('TER '):
                    outfile.write(line[:22] + nl[22:27] + '\n')
                else:
                    outfile.write(line)
    finally:
        junk = glob.glob(prefix + '*')
        for item in junk:
            item_filepath = os.path.abspath(item)
            os.remove(item_filepath)
