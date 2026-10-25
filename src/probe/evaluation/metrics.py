"""Public evaluation metrics with the paper's names and strict thresholds."""
from __future__ import annotations

import math
from numbers import Real
from rdkit.Chem import Descriptors, Lipinski
from probe.chemistry.molecules import chemical_metrics, molecule
from .reasonability import assess_reasonability


def success(vina_score, qed, sa):
    """Paper Success predicate, independent of MRR: Vina<-8.18, QED>.25, SA>.59."""
    for name, value in (('vina_score', vina_score), ('qed', qed), ('sa', sa)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite numeric score')
    return bool(vina_score < -8.18 and qed > .25 and sa > .59)


def lipinski_count(mol):
    """Count five molecular property criteria."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rules = (mw < 500, Lipinski.NumHDonors(mol) <= 5,
             Lipinski.NumHAcceptors(mol) <= 10, -2 <= logp <= 5,
             Lipinski.NumRotatableBonds(mol) <= 10)
    return int(sum(rules))


def score_molecule(smiles, mrr_mode='legacy'):
    """Compute 2D metrics; malformed or disconnected inputs raise ValueError."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('smiles must be a nonempty string')
    mol = molecule(smiles)
    properties = chemical_metrics(smiles)
    reasonability = assess_reasonability(mol, mode=mrr_mode)
    values = {**properties, 'mw': float(Descriptors.MolWt(mol)),
              'lipinski': lipinski_count(mol), 'mrr': bool(reasonability['reasonable']),
              'mrr_mode': mrr_mode, 'mrr_details': reasonability}
    for name in ('qed', 'sa', 'logp', 'mw'):
        if not math.isfinite(values[name]):
            raise ValueError(f'Nonfinite {name} computed for the molecule')
    return values
