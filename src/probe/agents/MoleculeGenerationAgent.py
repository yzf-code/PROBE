"""Shared fragment assembly engine for probing and iterative optimization."""
from __future__ import annotations

from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from probe.chemistry.fragments import FragmentLibrary, validate_constraints

ACTIONS = {'Grow', 'Delete_Group', 'Replace_Terminal_Group', 'Replace_Sidechain_or_Ring'}


class MoleculeGenerationAgent:
    def __init__(self, db_json_path, top_k=20):
        self.library = FragmentLibrary(db_json_path)
        self.top_k = top_k

    def _search_fragments(self, constraints, top_k=None):
        return self.library.search(constraints, self.top_k if top_k is None else top_k)

    @staticmethod
    def _release_carbon_hydrogens(atom):
        # Atom-mapped SMILES spell out [CH3:n]. Let RDKit recompute carbon H
        # after a graph edit rather than retaining that now-obsolete valence.
        if atom.GetAtomicNum() == 6:
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

    @staticmethod
    def _scaffold_anchor(mol, target_idx):
        """Choose a non-ring cut leaving the largest component on the scaffold side."""
        options = []
        for bond in mol.GetAtomWithIdx(target_idx).GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            neighbor = bond.GetOtherAtomIdx(target_idx)
            trial = Chem.RWMol(mol)
            trial.RemoveBond(target_idx, neighbor)
            components = Chem.GetMolFrags(trial.GetMol(), sanitizeFrags=False)
            kept = next((c for c in components if neighbor in c and target_idx not in c), None)
            if kept:
                options.append((len(kept), -neighbor, neighbor))
        if not options:
            raise ValueError('Target has no safe exocyclic single-bond cut; ring opening is unsupported')
        return max(options)[2]

    def _execute_surgery(self, original, target_idx, action, fragment_smiles=None):
        mol = Chem.Mol(original)
        rw = Chem.RWMol(mol)
        attachment_idx = target_idx
        if action != 'Grow':
            if action == 'Replace_Terminal_Group' and mol.GetAtomWithIdx(target_idx).GetDegree() != 1:
                raise ValueError('Replace_Terminal_Group requires a terminal atom; use Replace_Sidechain_or_Ring')
            anchor = self._scaffold_anchor(mol, target_idx)
            rw.RemoveBond(target_idx, anchor)
            components = Chem.GetMolFrags(rw.GetMol(), sanitizeFrags=False)
            kept = next(c for c in components if anchor in c)
            removed = set(range(mol.GetNumAtoms())) - set(kept)
            # Preserve indices using graph membership, not isotope marker 999.
            attachment_idx = anchor - sum(i < anchor for i in removed)
            for i in sorted(removed, reverse=True):
                rw.RemoveAtom(i)
            kept_anchor = rw.GetAtomWithIdx(attachment_idx)
            self._release_carbon_hydrogens(kept_anchor)
            kept_anchor.SetNoImplicit(False)
            kept_anchor.UpdatePropertyCache(strict=False)
        else:
            target = rw.GetAtomWithIdx(target_idx)
            if target.GetTotalNumHs() < 1:
                raise ValueError('Grow target has no hydrogen available; anchor is not changed automatically')

        if action == 'Delete_Group':
            result = rw.GetMol()
            Chem.SanitizeMol(result)
            return result

        fragment = Chem.MolFromSmiles(fragment_smiles)
        if fragment is None:
            raise ValueError('Invalid fragment SMILES')
        dummies = [a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
        if not dummies:
            raise ValueError('Fragment has no attachment point')
        first = fragment.GetAtomWithIdx(dummies[0])
        if first.GetDegree() != 1:
            raise ValueError('Invalid fragment attachment point')
        frag_anchor = first.GetNeighbors()[0].GetIdx()
        anchor_atom = rw.GetAtomWithIdx(attachment_idx)
        if action == 'Grow' and anchor_atom.GetNumExplicitHs():
            anchor_atom.SetNumExplicitHs(anchor_atom.GetNumExplicitHs() - 1)
        self._release_carbon_hydrogens(anchor_atom)
        offset = rw.GetNumAtoms()
        combined = Chem.RWMol(Chem.CombineMols(rw.GetMol(), fragment))
        combined.AddBond(attachment_idx, offset + frag_anchor, Chem.BondType.SINGLE)
        # Unused BRICS attachment points become H-capped, not free radicals.
        for dummy in dummies[1:]:
            atom = combined.GetAtomWithIdx(offset + dummy)
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
        combined.RemoveAtom(offset + dummies[0])
        result = Chem.RemoveHs(combined.GetMol(), sanitize=True)
        Chem.SanitizeMol(result)
        return result

    @staticmethod
    def _validate_product(mol, original_heavy_atoms):
        if mol.GetNumHeavyAtoms() < .5 * original_heavy_atoms:
            raise ValueError('Edit retains less than 50% of the original heavy atoms')
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError('Edit produced a disconnected molecule')
        if any(a.GetNumRadicalElectrons() or a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            raise ValueError('Edit left an uncapped attachment or radical')
        if (rdMolDescriptors.CalcNumRotatableBonds(mol) > 15 or
                Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRings(mol) > 6 or
                len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) > 5):
            raise ValueError('Product exceeds retained assembly limits (RotB 15, MW 800, rings 6, chiral centers 5)')

    def execute_editing(self, mapped_smi, mod_logic, used_smiles=None):
        """Return (canonical product SMILES, audit message), or (None, reason)."""
        used_smiles = set() if used_smiles is None else used_smiles
        try:
            if not isinstance(mod_logic, dict):
                raise ValueError('modification_logic must be an object')
            action = mod_logic.get('action')
            if action not in ACTIONS:
                raise ValueError(f'Unsupported action: {action!r}')
            targets = mod_logic.get('target_atom_indices')
            if not isinstance(targets, list) or len(targets) != 1 or type(targets[0]) is not int:
                raise ValueError('Exactly one integer atom-map target is required')
            constraints = mod_logic.get('constraints', {})
            validate_constraints(constraints)
            if not isinstance(mapped_smi, str) or not mapped_smi.strip():
                raise ValueError('mapped_smiles must be a nonempty string')
            mol = Chem.MolFromSmiles(mapped_smi)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError('Input must be a valid connected mapped molecule')
            atom_maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
            if any(i <= 0 for i in atom_maps) or len(set(atom_maps)) != len(atom_maps):
                raise ValueError('Every input atom must have a unique positive atom-map number')
            by_map = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
            if targets[0] not in by_map:
                raise ValueError(f'Unknown atom-map target {targets[0]}; no index fallback is allowed')
            index = by_map[targets[0]]
            # Keep the source itself out of the candidate set even if caller forgot.
            source = Chem.Mol(mol)
            for atom in source.GetAtoms(): atom.SetAtomMapNum(0)
            source_smiles = Chem.MolToSmiles(source)
            fragments = [None] if action == 'Delete_Group' else self._search_fragments(constraints)
            smarts = mod_logic.get('pharmacophore_smarts', '')
            if not isinstance(smarts, str):
                raise ValueError('pharmacophore_smarts must be a string')
            if smarts:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None: raise ValueError('Invalid pharmacophore_smarts')
                fragments = [s for s in fragments if s and Chem.MolFromSmiles(s).HasSubstructMatch(pattern)]
            if not fragments:
                raise ValueError('No fragments match the strict semantic constraints; no constraints were relaxed')
            failures = Counter()
            for fragment in fragments:
                try:
                    product = self._execute_surgery(mol, index, action, fragment)
                    self._validate_product(product, mol.GetNumHeavyAtoms())
                    for atom in product.GetAtoms(): atom.SetAtomMapNum(0)
                    smiles = Chem.MolToSmiles(product, isomericSmiles=True)
                    if smiles == source_smiles or smiles in used_smiles:
                        raise ValueError('Duplicate or unchanged product')
                    used_smiles.add(smiles)
                    return smiles, f'Success: {action} at atom-map {targets[0]}; fragment={fragment}; strict retrieval'
                except (ValueError, RuntimeError) as exc:
                    failures[str(exc)] += 1
            return None, 'All retrieved fragments failed: ' + '; '.join(f'{n} x {why}' for why, n in failures.items())
        except (ValueError, RuntimeError) as exc:
            return None, str(exc)
