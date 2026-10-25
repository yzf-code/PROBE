"""BRICS library construction and semantic retrieval (no bundled compound data)."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors, QED, rdMolDescriptors
from .molecules import chemical_metrics

TAGS = {
    'size': {'Small', 'Medium', 'Large', 'Any'},
    'polarity': {'Hydrophilic', 'Neutral', 'Lipophilic', 'Any'},
    'flexibility': {'Rigid', 'Flexible', 'Any'},
    'shape': {'Disc-like', 'Rod-like', 'Sphere-like', 'Any'},
    'charge': {0, 1, -1, '0', '+1', '1', '-1', 'Any'},
}
logger = logging.getLogger(__name__)

REQUIRED = {'smiles', 'MW', 'LogP', 'TPSA', 'RotatableBonds', 'FormalCharge',
            'shape_category', 'QED', 'SA', 'ChiralCenters'}


def validate_constraints(constraints):
    if not isinstance(constraints, dict):
        raise ValueError('constraints must be a JSON object')
    unknown = set(constraints) - set(TAGS)
    if unknown:
        raise ValueError(f'Unknown semantic axes: {sorted(unknown)}')
    for axis, value in constraints.items():
        if not isinstance(value, (str, int)) or isinstance(value, bool) or value not in TAGS[axis]:
            raise ValueError(f'Invalid semantic constraint {axis}={value!r}')


class FragmentLibrary:
    """Strict, typed filtering; constraints are never evaluated as Python code."""
    def __init__(self, path):
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(rows, list) or not rows:
            raise ValueError('Fragment database must be a nonempty JSON array of records')
        self.total_count = len(rows)
        self.rejected_entries = []
        self.rows = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or REQUIRED - row.keys():
                raise ValueError(f'Fragment record {i} is missing required property columns')
            for key in REQUIRED - {'smiles', 'shape_category'}:
                if not isinstance(row[key], (int, float)) or isinstance(row[key], bool) or not math.isfinite(row[key]):
                    raise ValueError(f'Fragment record {i}: {key} must be finite numeric data')
            if not 0 <= row['QED'] <= 1 or not 0 <= row['SA'] <= 1:
                raise ValueError(f'Fragment record {i}: QED and normalized SA must be in [0, 1]')
            if not isinstance(row['smiles'], str):
                raise ValueError(f'Fragment record {i}: smiles must be a string')
            mol = Chem.MolFromSmiles(row['smiles'])
            reason = None
            if mol is None:
                reason = 'Invalid fragment SMILES'
            elif len(Chem.GetMolFrags(mol)) != 1:
                reason = 'Disconnected fragment cannot be assembled as one attachment'
            else:
                dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
                if not dummies:
                    reason = 'No BRICS dummy attachment point'
                elif any(a.GetDegree() != 1 for a in dummies):
                    reason = 'Dummy attachment point must have exactly one neighbor'
                elif mol.GetNumHeavyAtoms() == 0:
                    reason = 'Fragment has no heavy atoms'
            if reason:
                self.rejected_entries.append({'index': i, 'reason': reason})
            else:
                self.rows.append(row)
        self.usable_count = len(self.rows)
        if self.rejected_entries:
            logger.warning(
                'Fragment library: rejected %d of %d records as non-assemblable; '
                '%d usable. Inspect FragmentLibrary.rejected_entries for indices and reasons.',
                len(self.rejected_entries), self.total_count, self.usable_count,
            )
        if not self.rows:
            raise ValueError(f'Fragment database contains no usable attachable records '
                             f'({self.total_count} rejected); first rejection: '
                             f'{self.rejected_entries[0]["reason"]}')

    def search(self, constraints, top_k=20):
        validate_constraints(constraints)
        if top_k < 1:
            raise ValueError('top_k must be positive')
        def matches(row):
            size = constraints.get('size', 'Any')
            if size == 'Small' and not row['MW'] < 150: return False
            if size == 'Medium' and not 150 <= row['MW'] <= 250: return False
            if size == 'Large' and not row['MW'] > 250: return False
            polarity = constraints.get('polarity', 'Any')
            if polarity == 'Hydrophilic' and not (row['LogP'] <= 1.5 and row['TPSA'] >= 40): return False
            if polarity == 'Lipophilic' and not (row['LogP'] >= 2.5 and row['TPSA'] <= 40): return False
            if polarity == 'Neutral' and not 1.5 < row['LogP'] < 2.5: return False
            flex = constraints.get('flexibility', 'Any')
            if flex == 'Rigid' and row['RotatableBonds'] > 1: return False
            if flex == 'Flexible' and row['RotatableBonds'] <= 1: return False
            charge = constraints.get('charge', 'Any')
            if charge != 'Any' and row['FormalCharge'] != int(charge): return False
            shape = constraints.get('shape', 'Any')
            return shape == 'Any' or row['shape_category'] == shape
        rows = [r for r in self.rows if matches(r)]
        rows.sort(key=lambda r: (-(.5*r['QED'] + .5*r['SA'] - .15*r['ChiralCenters']), r['smiles']))
        return [r['smiles'] for r in rows[:top_k]]


def cap_dummies(mol):
    """Replace BRICS attachment dummies with H for descriptor calculation."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    capped = Chem.RemoveHs(rw.GetMol(), sanitize=True)
    Chem.SanitizeMol(capped)
    if not capped.GetNumHeavyAtoms():
        raise ValueError('Fragment contains no heavy atoms')
    return capped


def annotate_fragment(smiles, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('Expected a BRICS fragment SMILES containing attachment dummies')
    capped = cap_dummies(mol)
    metrics = chemical_metrics(Chem.MolToSmiles(capped))
    mol3d = Chem.AddHs(capped)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol3d, params) != 0:
        raise ValueError('Cannot embed fragment to compute its shape category')
    npr1, npr2 = rdMolDescriptors.CalcNPR1(mol3d), rdMolDescriptors.CalcNPR2(mol3d)
    shape = 'Intermediate'
    if npr1 <= .25 and npr2 >= .75: shape = 'Rod-like'
    elif npr1 >= .75 and npr2 >= .75: shape = 'Sphere-like'
    elif .25 <= npr1 <= .75 and npr2 <= .75: shape = 'Disc-like'
    return {
        'smiles': Chem.MolToSmiles(mol), 'MW': Descriptors.MolWt(capped),
        'LogP': Descriptors.MolLogP(capped), 'TPSA': Descriptors.TPSA(capped),
        'RotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(capped),
        'FormalCharge': Chem.GetFormalCharge(capped), 'shape_category': shape,
        'QED': metrics['qed'], 'SA': metrics['sa'],
        'ChiralCenters': len(Chem.FindMolChiralCenters(capped, includeUnassigned=True)),
        'num_attachments': sum(a.GetAtomicNum() == 0 for a in mol.GetAtoms()),
        'NPR1': npr1, 'NPR2': npr2,
    }


def build_library(input_path, output_path, *, input_is_fragments=False, seed=42):
    """Read SMILES [optional name] lines; decompose whole compounds or annotate fragments."""
    fragments, rejected = set(), []
    for line_number, line in enumerate(Path(input_path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'): continue
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected.append({'line': line_number, 'error': 'Invalid SMILES'})
            continue
        if input_is_fragments:
            fragments.add(Chem.MolToSmiles(mol))
        else:
            fragments.update(BRICS.BRICSDecompose(mol))
    records = []
    for smiles in sorted(fragments):
        if '*' not in smiles: continue
        try:
            records.append(annotate_fragment(smiles, seed=seed))
        except (ValueError, RuntimeError) as exc:
            rejected.append({'fragment': smiles, 'error': str(exc)})
    if not records:
        raise ValueError('No valid attachable fragments were produced')
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    return {'written': len(records), 'rejected': rejected}
