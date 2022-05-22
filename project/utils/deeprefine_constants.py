from typing import Dict, List

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from Graphein (https://www.github.com/a-r-j/graphein):
# -------------------------------------------------------------------------------------------------------------------------------------
# https://www.ebi.ac.uk/pdbe-srv/pdbechem/chemicalCompound
RESI_THREE_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ASX": "B",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "GLX": "Z",
    "CSD": "C",
    "HYP": "P",
    "BMT": "T",
    "3HP": "X",
    "4HP": "X",
    "5HP": "Q",
    "ACE": "X",
    "ABA": "A",
    "AIB": "A",
    "NH2": "X",
    "CBX": "X",
    "CSW": "C",
    "OCS": "C",
    "DAL": "A",
    "DAR": "R",
    "DSG": "N",
    "DSP": "D",
    "DCY": "C",
    "CRO": "TYG",
    "DGL": "E",
    "DGN": "Q",
    "DHI": "H",
    "DIL": "I",
    "DIV": "V",
    "DLE": "L",
    "DLY": "K",
    "DPN": "F",
    "DPR": "P",
    "DSN": "S",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DVA": "V",
    "FOR": "X",
    "CGU": "E",
    "IVA": "X",
    "KCX": "K",
    "LLP": "K",
    "CXM": "M",
    "FME": "M",
    "MLE": "L",
    "MVA": "V",
    "NLE": "L",
    "PTR": "Y",
    "ORN": "A",
    "SEP": "S",
    "SEC": "U",
    "TPO": "T",
    "PCA": "Q",
    "PVL": "X",
    "PYL": "O",
    "SAR": "G",
    "CEA": "C",
    "CSO": "C",
    "CSS": "C",
    "CSX": "C",
    "CME": "C",
    "TYS": "Y",
    "BOC": "X",
    "TPQ": "Y",
    "STY": "Y",
    "UNK": "X",
}
"""
Mapping of 3-letter residue names to 1-letter residue names.
Non-standard/modified amino acids are mapped to their parent amino acid.
Includes ``"UNK"`` to denote unknown residues.
"""

BASE_AMINO_ACIDS: List[str] = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
"""Vocabulary of 20 standard amino acids."""

# Atom classes based on Heyrovska, Raji covalent radii paper.
DEFAULT_BOND_STATE: Dict[str, str] = {"N": "Nsb", "CA": "Csb", "C": "Cdb", "O": "Odb", "OXT": "Osb", "CB": "Csb",
                                      "H": "Hsb",
                                      # Not sure about these - assuming they're all standard Hydrogen. Won't make much difference given
                                      # the tolerance is larger than Hs covalent radius
                                      "HG1": "Hsb", "HE": "Hsb", "1HH1": "Hsb", "1HH2": "Hsb", "2HH1": "Hsb",
                                      "2HH2": "Hsb", "HG": "Hsb", "HH": "Hsb", "1HD2": "Hsb", "2HD2": "Hsb",
                                      "HZ1": "Hsb", "HZ2": "Hsb", "HZ3": "Hsb", }

RESIDUE_ATOM_BOND_STATE: Dict[str, Dict[str, str]] = {
    "XXX": {"N": "Nsb", "CA": "Csb", "C": "Cdb", "O": "Odb", "OXT": "Osb", "CB": "Csb", "H": "Hsb", },
    "VAL": {"CG1": "Csb", "CG2": "Csb"}, "LEU": {"CG": "Csb", "CD1": "Csb", "CD2": "Csb"},
    "ILE": {"CG1": "Csb", "CG2": "Csb", "CD1": "Csb"}, "MET": {"CG": "Csb", "SD": "Ssb", "CE": "Csb"},
    "PHE": {"CG": "Cdb", "CD1": "Cres", "CD2": "Cres", "CE1": "Cdb", "CE2": "Cdb", "CZ": "Cres", },
    "PRO": {"CG": "Csb", "CD": "Csb"}, "SER": {"OG": "Osb"}, "THR": {"OG1": "Osb", "CG2": "Csb"}, "CYS": {"SG": "Ssb"},
    "ASN": {"CG": "Csb", "OD1": "Odb", "ND2": "Ndb"}, "GLN": {"CG": "Csb", "CD": "Csb", "OE1": "Odb", "NE2": "Ndb"},
    "TYR": {"CG": "Cdb", "CD1": "Cres", "CD2": "Cres", "CE1": "Cdb", "CE2": "Cdb", "CZ": "Cres", "OH": "Osb", },
    "TRP": {"CG": "Cdb", "CD1": "Cdb", "CD2": "Cres", "NE1": "Nsb", "CE2": "Cdb", "CE3": "Cdb", "CZ2": "Cres",
            "CZ3": "Cres", "CH2": "Cdb", }, "ASP": {"CG": "Csb", "OD1": "Ores", "OD2": "Ores"},
    "GLU": {"CG": "Csb", "CD": "Csb", "OE1": "Ores", "OE2": "Ores"},
    "HIS": {"CG": "Cdb", "CD2": "Cdb", "ND1": "Nsb", "CE1": "Cdb", "NE2": "Ndb", },
    "LYS": {"CG": "Csb", "CD": "Csb", "CE": "Csb", "NZ": "Nsb"},
    "ARG": {"CG": "Csb", "CD": "Csb", "NE": "Nsb", "CZ": "Cdb", "NH1": "Nres", "NH2": "Nres", }, }

# Covalent radii from Heyrovska, Raji : 'Atomic Structures of all the Twenty
# Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic
# Covalent Radii' <https://arxiv.org/pdf/0804.2488.pdf>
# Adding Ores between Osb and Odb for Asp and Glu, Nres between Nsb and Ndb
# for Arg, as PDB does not specify

COVALENT_RADII: Dict[str, float] = {"Csb": 0.77, "Cres": 0.72, "Cdb": 0.67, "Osb": 0.67, "Ores": 0.635, "Odb": 0.60,
                                    "Nsb": 0.70, "Nres": 0.66, "Ndb": 0.62, "Hsb": 0.37, "Ssb": 1.04, }

COVALENT_RADIUS_TOLERANCE = 0.56  # 0.4, 0.45, or 0.56 - These are common distance tolerances for covalent bonds

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
PROT_ATOM_NAMES = ['N', 'CA', 'C', 'O', 'CB', 'OG', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OD1', 'ND2', 'CG1', 'CG2',
                   'CD', 'CE', 'NZ', 'OD2', 'OE1', 'NE2', 'OE2', 'OH', 'NE', 'NH1', 'NH2', 'OG1', 'SD', 'ND1', 'SG',
                   'NE1', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT', 'UNX']  # 'UNX' represents the unknown atom type

# Dataset-global maximum constant for feature normalization
MAX_FEATS_CONST = 100.0

# Dihedral angle mapping
DIHEDRAL_ANGLE_ID_TO_ATOM_NAMES = {
    0: ['N', 'CA', 'C', 'N'],
    1: ['CA', 'C', 'N', 'CA'],
    2: ['C', 'N', 'CA', 'C']
}


def get_allowable_feats(ca_only: bool):
    return [BASE_AMINO_ACIDS, ] if ca_only else [PROT_ATOM_NAMES, ]
