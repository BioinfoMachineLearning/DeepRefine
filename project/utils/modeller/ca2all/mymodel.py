import logging

try:
    from modeller import *
    from modeller.automodel import *
except:
    logging.info(f'mymodel.py: Modeller not installed')


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Modeller at: https://bitbucket.org/lcbio/ca2all/src/master/
# -------------------------------------------------------------------------------------------------------------------------------------

class MyModel(automodel):
    def select_atoms(self):
        return selection(self) - selection(self).only_atom_types('CA')
