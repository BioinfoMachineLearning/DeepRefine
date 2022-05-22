import argparse
import logging

try:
    from modeller import *
except:
    logging.info(f'NumChains.py: Modeller not installed')


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Modeller at: https://bitbucket.org/lcbio/ca2all/src/master/
# -------------------------------------------------------------------------------------------------------------------------------------

# This demonstrates how to renumber residues in the chain starting from 1 (chain breaks are kept).
def NumChains(filename):
    env = environ()
    env.io.atom_files_directory = ['.']
    mdl = model(env)
    mdl.read(filename)

    # Renumber all residues in the new chain starting from 1
    K = 0
    P = 0
    C = []
    for c in mdl.chains:
        K = int(K) + 1
        C += c.name
        for num, residue in enumerate(mdl.chains[c.name].residues):
            if K == 1:
                if num == 0:
                    N = residue.num
                residue.num = '%d' % (int(residue.num) - int(N) + 1)
                P = residue.num
            else:
                residue.num = '%d' % (int(P) + 1)
                P = int(P) + 1

    mdl.write(filename, model_format='PDB')
    return [1, C[0], P, C[len(C) - 1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NumChains',
        description="""""",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=''
    )
    parser.add_argument(
        '-i', '--input-pdb',
        help='input pdb file with structure',
        metavar='INPUT',
        dest='inp',
        required=True
    )
    args = parser.parse_args()
    NumChains(args.inp)
