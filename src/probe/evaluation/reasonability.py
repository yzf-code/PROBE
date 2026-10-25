"""Molecular reasonability checks."""
from __future__ import annotations

from collections import defaultdict
from rdkit import Chem

MRR_MODES = ('legacy', 'corrected')


def carbonyl_carbon_indices(mol):
    """Return C=O and C=N carbon indices for the hybridization check."""
    indices = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        if first.GetSymbol() == 'C' and second.GetSymbol() in ('N', 'O'):
            indices.add(first.GetIdx())
        elif second.GetSymbol() == 'C' and first.GetSymbol() in ('N', 'O'):
            indices.add(second.GetIdx())
    return indices


def ring_groups(mol):
    """Group RDKit rings connected by any shared atom."""
    rings = mol.GetRingInfo().AtomRings()
    assigned = set()
    groups = []
    for start in range(len(rings)):
        if start in assigned:
            continue
        assigned.add(start)
        group, stack = [rings[start]], [start]
        while stack:
            current = stack.pop()
            for index, ring in enumerate(rings):
                if index not in assigned and set(rings[current]).intersection(ring):
                    assigned.add(index)
                    group.append(ring)
                    stack.append(index)
        groups.append(tuple(group))
    return groups


def assess_reasonability(mol, mode='legacy'):
    """Return the per-molecule MRR decision with inspectable diagnostics."""
    if mode not in MRR_MODES:
        raise ValueError(f'mrr_mode must be one of {MRR_MODES}')
    if mol is None or not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() == 0:
        raise ValueError('MRR requires a valid nonempty RDKit molecule')
    groups = ring_groups(mol)
    co_ids = carbonyl_carbon_indices(mol)
    ring_atoms = {i for group in groups for ring in group for i in ring}
    memberships = defaultdict(set)
    for group in groups:
        for ring in group:
            for i in ring:
                memberships[i].add(tuple(ring))
    details = {
        'mode': mode, 'reasonable': True,
        'ring_atom_ratio': len(ring_atoms)/mol.GetNumAtoms(),
        'unreasonable_atom_ratio': 0.0,
        'shared_atom_count': sum(len(rings) > 1 for rings in memberships.values()),
        'ring_system_count': len(groups), 'atom_index_base': 0,
        'carbonyl_carbon_indices': sorted(co_ids), 'violations': [],
        'ratio_note': 'Legacy sum-of-ring-sizes numerator can double-count shared atoms and exceed one.',
    }
    multiring = [g for g in groups if len(g) > 1]

    def contains_sp2_carbon(group):
        return any(mol.GetAtomWithIdx(i).GetSymbol() == 'C' and
                   mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2
                   for ring in group for i in ring)

    if mode == 'legacy':
        failed_groups = multiring if multiring and not any(contains_sp2_carbon(g) for g in multiring) else []
    else:
        failed_groups = [g for g in multiring if not contains_sp2_carbon(g)]
    if failed_groups:
        details['reasonable'] = False
        details['unreasonable_atom_ratio'] = sum(len(r) for g in failed_groups for r in g)/len(ring_atoms)
        details['violations'] = [{'rule': 'all_sp3_multiring_system',
                                  'rings': [list(r) for r in group]} for group in failed_groups]
        return details

    for group in groups:
        verified_sp2, verified_non_sp2 = set(), set()
        remaining_rings = []
        for ring in group:
            sp2_atoms, non_sp2_atoms = [], []
            for i in ring:
                atom = mol.GetAtomWithIdx(i)
                if atom.GetSymbol() == 'C' and i not in co_ids:
                    if atom.GetHybridization() == Chem.HybridizationType.SP2:
                        sp2_atoms.append(i)
                    else:
                        non_sp2_atoms.append(i)
            if len(sp2_atoms) == len(ring):
                verified_sp2.update(ring)
            elif len(non_sp2_atoms) == len(ring):
                verified_non_sp2.update(ring)
            else:
                remaining_rings.append(ring)
        while remaining_rings:
            new_sp2, new_non_sp2 = set(), set()
            for ring in remaining_rings[:]:
                remaining_atoms = set(ring)-verified_sp2-verified_non_sp2
                eligible = [i for i in remaining_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C' and i not in co_ids]
                consistently_sp2 = all(mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2 for i in eligible)
                consistently_non_sp2 = all(mol.GetAtomWithIdx(i).GetHybridization() != Chem.HybridizationType.SP2 for i in eligible)
                if consistently_sp2:
                    new_sp2.update(remaining_atoms)
                    remaining_rings.remove(ring)
                elif consistently_non_sp2:
                    new_non_sp2.update(remaining_atoms)
                    remaining_rings.remove(ring)
            verified_sp2.update(new_sp2)
            verified_non_sp2.update(new_non_sp2)
            if not new_sp2 and not new_non_sp2:
                break
        if remaining_rings:
            details['reasonable'] = False
            details['unreasonable_atom_ratio'] = sum(len(r) for r in remaining_rings)/len(ring_atoms)
            details['violations'] = [{'rule': 'unresolved_mixed_hybridization',
                                      'rings': [list(r) for r in remaining_rings]}]
            break
    return details
