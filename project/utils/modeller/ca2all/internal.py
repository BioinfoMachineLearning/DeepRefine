import logging
import sys
from os.path import basename

try:
    from modeller import *
    from modeller.automodel import *
    from Align import Align
    from mymodel import MyModel
except:
    logging.info(f'internal.py: Modeller not installed')


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Modeller at: https://bitbucket.org/lcbio/ca2all/src/master/
# -------------------------------------------------------------------------------------------------------------------------------------


def Rebuild(filename, pir, template=None, output=None, iterations=1):
    prefix = basename(filename).rsplit('.', 1)[0]

    #####---MODELLER SECTION---#####
    env = environ()
    env.io.atom_files_directory = ['.']
    env.libs.topology.read(file='$(LIB)/top_allh.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')

    ###--STAGE_1: frozen CA-trace; restraints from CA-trace and template
    mdl = MyModel(
        env,
        alnfile=pir,
        knowns=('initial', 'template'),
        sequence=prefix,
        inifile=filename,
        assess_methods=assess.DOPE
    )
    mdl.starting_model = 1
    mdl.ending_model = int(iterations)
    mdl.md_level = refine.fast
    mdl.auto_align(matrix_file=prefix + '.mat')
    mdl.final_malign3d = True
    mdl.make()
    models = [m for m in mdl.outputs if m['failure'] is None]
    cmp_key = 'DOPE score'
    models.sort(lambda x, y: cmp(x[cmp_key], y[cmp_key]))
    final = models[0]['name'].rsplit('.', 1)[0] + '_fit.pdb'

    ###--STAGE_2: flexible modeling (all atoms, also CA-trace) based on the structure from stage_1
    Align(filename, final, final, pir)
    new = automodel(
        env,
        alnfile=pir,
        knowns=('initial'),
        sequence=models[0]['name'].rsplit('.', 1)[0] + '_fit',
        inifile=final,
        assess_methods=assess.DOPE
    )
    new.md_level = refine.fast
    new.make()
    new.write(final)

    return final


def NumResidues(final, atoms, output):
    ####---Residues renumber SECTION---####
    if output:
        outfile = open(output, 'w')
    else:
        outfile = sys.stdout
    with open(final) as f:
        a = iter(atoms)  # structure of atoms is: ('GLU', 'A', ' 207 ', '  1.00 37.96')
        current = ch = r = t = nl = None
        for line in f:
            if line.startswith('ATOM'):
                res = line[21:27]
                if not current or current != res:
                    current = res
                    ch, r, t = a.next()[1:]
                nl = line[:21] + ch + r + line[27:54] + t
                if len(line) > 66:
                    nl += line[66:]
                outfile.write(nl)
            elif line.startswith('TER '):
                outfile.write(line[:22] + nl[22:27] + '\n')
            else:
                outfile.write(line)
