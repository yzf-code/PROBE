"""Fragment-resolved PLIP, geometry and medicinal-chemistry signals."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors, GraphDescriptors, rdMolDescriptors

RADII = {'H': 1.10, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
         'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}


def _aligned_pose(smiles, pose_path):
    original = Chem.MolFromSmiles(smiles)
    pose = Chem.MolFromMolFile(str(pose_path), removeHs=True)
    if original is None or pose is None or not pose.GetNumConformers():
        raise ValueError('Invalid molecule or missing docked pose coordinates')
    original = Chem.RemoveHs(original)
    for mol in (original, pose):
        for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    match = pose.GetSubstructMatch(original, useChirality=True)
    if original.GetNumAtoms() != pose.GetNumAtoms() or len(match) != original.GetNumAtoms():
        raise ValueError('Docked pose graph does not match the input molecule; no positional mapping fallback')
    pose = Chem.RenumberAtoms(pose, list(match))
    for atom in pose.GetAtoms(): atom.SetAtomMapNum(atom.GetIdx()+1)
    return original, pose


def fragment_records(mol):
    """Disjoint, contiguous BRICS fragments with their original map-number sets."""
    cuts = [mol.GetBondBetweenAtoms(a, b).GetIdx() for (a, b), _ in BRICS.FindBRICSBonds(mol)]
    broken = Chem.FragmentOnBonds(mol, cuts, addDummies=False) if cuts else Chem.Mol(mol)
    records = []
    for n, indices in enumerate(Chem.GetMolFrags(broken, sanitizeFrags=False), 1):
        records.append({'fragment_name': f'BRICS_{n}',
                        'atom_indices': [i+1 for i in indices],
                        'fragment_smiles': Chem.MolFragmentToSmiles(mol, atomsToUse=list(indices), canonical=True)})
    return records


def _medicinal_signals(mol, score):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    alerts = []
    spiro, bridge = rdMolDescriptors.CalcNumSpiroAtoms(mol), rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    if spiro: alerts.append(f'{spiro} spiro atoms: synthesis-complexity liability')
    if bridge: alerts.append(f'{bridge} bridgehead atoms: synthesis-complexity liability')
    macro = sum(len(r) > 8 for r in rings)
    if macro: alerts.append(f'{macro} macrocycles with more than 8 atoms')
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    if n_chiral > 2: alerts.append(f'{n_chiral} chiral centers: synthesis-complexity liability')
    remaining = list(rings)
    while remaining:
        group = [remaining.pop()]
        changed = True
        while changed:
            changed = False
            for ring in remaining[:]:
                if any(ring & r for r in group):
                    remaining.remove(ring); group.append(ring); changed = True
        if len(group) > 1 and not any(mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2
                                     for i in set().union(*group)):
            alerts.append('All-sp3 polycyclic ring system (paper structural-alert heuristic)')
    exempt = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            if a.GetAtomicNum() == 6 and b.GetAtomicNum() in (7, 8): exempt.add(a.GetIdx())
            if b.GetAtomicNum() == 6 and a.GetAtomicNum() in (7, 8): exempt.add(b.GetIdx())
    for ring in rings:
        carbon = [mol.GetAtomWithIdx(i) for i in ring if i not in exempt and mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
        sp2 = sum(a.GetHybridization() == Chem.HybridizationType.SP2 for a in carbon)
        if 0 < sp2 < len(carbon): alerts.append('Mixed sp2/sp3 ring carbon hybridization (paper heuristic)')
    patterns = {'Aniline': 'c1ccccc1[NX3;H2,H1;!$(NC=O)]', 'Michael acceptor': '[C,c]=[C,c]C(=O)',
                'Hydrazine/azo': '[N,n][N,n]', 'Aldehyde': '[CH]=O'}
    for name, smarts in patterns.items():
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            alerts.append(f'{name} at atom maps {[i+1 for i in match]}')
    data = {'le': float(score)/mol.GetNumHeavyAtoms(), 'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol), 'tpsa': Descriptors.TPSA(mol),
            'chiral_centers': n_chiral, 'sa_proxy': GraphDescriptors.BertzCT(mol), 'alerts': alerts}
    return json.dumps(data, indent=2), data


def _geometry_signals(receptor_path, ligand):
    receptor = Chem.MolFromPDBFile(str(receptor_path), sanitize=False, removeHs=False)
    if receptor is None or not receptor.GetNumConformers():
        raise ValueError('Cannot read receptor coordinates for geometry analysis')
    coords = receptor.GetConformer().GetPositions()
    lig_coords = ligand.GetConformer().GetPositions()
    if not np.isfinite(coords).all() or not np.isfinite(lig_coords).all():
        raise ValueError('Geometry input contains nonfinite coordinates')
    AllChem.ComputeGasteigerCharges(ligand)
    info = []
    for atom in receptor.GetAtoms():
        residue = atom.GetPDBResidueInfo()
        name = residue.GetResidueName().strip().upper() if residue else 'UNK'
        label = f'{name}{residue.GetResidueNumber()}:{residue.GetChainId().strip()}' if residue else 'UNKNOWN'
        symbol = atom.GetSymbol()
        charge = 'negative' if name in ('ASP', 'GLU') and symbol == 'O' else (
            'positive' if name in ('LYS', 'ARG', 'HIS') and symbol == 'N' else 'neutral')
        info.append((symbol, label, charge))
    clashes, voids, mismatch = [], [], []
    for atom in ligand.GetAtoms():
        i, symbol = atom.GetIdx(), atom.GetSymbol()
        atom_id = f'{symbol}{i+1}'
        distances = np.linalg.norm(coords-lig_coords[i], axis=1)
        nearest = float(distances.min())
        if nearest > 4.5: voids.append(f'Atom {atom_id} exposed/dangling; nearest receptor atom {nearest:.2f} A')
        contacts = np.flatnonzero(distances < 4.5)
        for j in np.flatnonzero(distances < 4.0):
            psym, label, _ = info[j]
            if psym == 'H': continue
            ratio = .6 if symbol in ('N', 'O') and psym in ('N', 'O') else .75
            if distances[j] < ratio*(RADII.get(symbol, 1.7)+RADII.get(psym, 1.7)):
                clashes.append(f'Atom {atom_id} clashes with {label}; distance {distances[j]:.2f} A')
        charge = float(atom.GetProp('_GasteigerCharge'))
        if not np.isfinite(charge): raise ValueError(f'Nonfinite Gasteiger charge at atom-map {i+1}')
        if abs(charge) > .2:
            charge_type = 'positive' if charge > 0 else 'negative'
            for j in contacts:
                if distances[j] < 4.0 and info[j][2] == charge_type:
                    mismatch.append(f'Atom {atom_id}: electrostatic repulsion near {info[j][1]} ({distances[j]:.2f} A)')
                    break
        if symbol in ('N', 'O') and len(contacts) >= 3:
            partner = any(distances[j] < 3.5 and info[j][0] in ('N', 'O') for j in contacts)
            if not partner:
                mismatch.append(f'Atom {atom_id}: buried polar atom with no N/O partner within 3.5 A')
    return ('\n'.join(clashes) or 'No significant steric clashes.',
            '\n'.join(voids) or 'No exposed atoms under the 4.5 A heuristic.',
            '\n'.join(mismatch) or 'No chemical mismatch under the stated heuristics.')


def _complex_pdb(receptor_path, ligand, output_path):
    # Reassign unique consecutive PDB serials. PLIP report *_orig_idx fields
    # refer to these serials, not to RDKit indices.
    records = [l for l in Path(receptor_path).read_text().splitlines() if l.startswith(('ATOM  ', 'HETATM'))]
    used_chains = {l[21] for l in records}
    chain = next((c for c in 'ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210' if c not in used_chains), None)
    if chain is None: raise ValueError('No spare PDB chain for the docked ligand')
    ligand = Chem.AddHs(ligand, addCoords=True)
    if len(records)+ligand.GetNumAtoms() > 99999: raise ValueError('PDB serial limit exceeded')
    lines = [f'{line[:6]}{n:5d}{line[11:]}' for n, line in enumerate(records, 1)]
    serial_to_map = {}
    for atom in ligand.GetAtoms():
        residue = Chem.AtomPDBResidueInfo()
        residue.SetName(f'{atom.GetSymbol()}{atom.GetIdx()+1}'[:4].rjust(4))
        residue.SetResidueName('UNL'); residue.SetResidueNumber(1)
        residue.SetChainId(chain); residue.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(residue)
        if atom.GetAtomicNum() > 1:
            serial_to_map[len(records)+atom.GetIdx()+1] = atom.GetAtomMapNum()
    for line in Chem.MolToPDBBlock(ligand).splitlines():
        if line.startswith(('ATOM  ', 'HETATM')):
            old = int(line[6:11]); lines.append(f'{line[:6]}{old+len(records):5d}{line[11:]}')
        elif line.startswith('CONECT'):
            nums = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
            lines.append('CONECT'+''.join(f'{x+len(records):5d}' for x in nums))
    Path(output_path).write_text('\n'.join(lines)+'\nEND\n')
    return chain, serial_to_map


def _plip_signals(receptor_path, ligand, folder, fragments):
    try:
        from plip.structure.preparation import PDBComplex
        from plip.exchange.report import BindingSiteReport
    except ImportError as exc:
        raise RuntimeError('Signal extraction requires PLIP and its OpenBabel Python bindings') from exc
    complex_path = folder/'complex.pdb'
    chain, serial_map = _complex_pdb(receptor_path, ligand, complex_path)
    profiler = PDBComplex()
    profiler.output_path = str(folder)
    profiler.load_pdb(str(complex_path))
    targets = [l for l in profiler.ligands if l.hetid == 'UNL' and l.chain == chain and l.position == 1]
    if len(targets) != 1:
        raise RuntimeError('PLIP did not identify the docked ligand uniquely')
    profiler.characterize_complex(targets[0])
    key = f'UNL:{chain}:1'
    if key not in profiler.interaction_sets:
        raise RuntimeError('PLIP produced no binding-site analysis for the docked ligand')
    report = BindingSiteReport(profiler.interaction_sets[key])
    by_atom = {i: f['fragment_name'] for f in fragments for i in f['atom_indices']}
    interactions = []
    for kind in ('hydrophobic', 'hbond', 'waterbridge', 'saltbridge', 'pistacking', 'pication', 'halogen', 'metal'):
        if not hasattr(report, kind+'_features') or not hasattr(report, kind+'_info'):
            raise RuntimeError(f'Unsupported PLIP report schema for {kind}')
        features = getattr(report, kind+'_features')
        for values in getattr(report, kind+'_info'):
            row = dict(zip(features, values))
            atom_ids = []
            if 'LIG_IDX_LIST' in row: atom_ids = [int(x) for x in str(row['LIG_IDX_LIST']).split(',')]
            elif 'LIGCARBONIDX' in row: atom_ids = [int(row['LIGCARBONIDX'])]
            elif kind in ('hbond', 'waterbridge'):
                suffix = '_IDX' if kind == 'waterbridge' else 'IDX'
                atom_ids = [int(row[('ACCEPTOR' if row['PROTISDON'] else 'DONOR')+suffix])]
            elif kind == 'halogen': atom_ids = [int(row['DON_IDX'])]
            elif kind == 'metal' and row.get('LOCATION') == 'ligand': atom_ids = [int(row['TARGET_IDX'])]
            if not atom_ids: continue
            unknown = [i for i in atom_ids if i not in serial_map]
            if unknown: raise RuntimeError(f'PLIP ligand serials cannot be mapped unambiguously: {unknown}')
            maps = [serial_map[i] for i in atom_ids]
            residue = f"{row.get('RESTYPE','?')}{row.get('RESNR','?')}:{row.get('RESCHAIN','?')}"
            interactions.append({'interaction': kind, 'atom_maps': maps,
                                 'fragments': sorted({by_atom[i] for i in maps}), 'residue': residue,
                                 'distance': row.get('DIST', row.get('DIST_D-A', row.get('CENTDIST')))})
    return interactions


def extract_signals(receptor_pdb, pose_path, smiles, score):
    original, pose = _aligned_pose(smiles, pose_path)
    fragments = fragment_records(original)
    folder = Path(tempfile.mkdtemp(prefix='signals-', dir=Path(pose_path).parent))
    interactions = _plip_signals(receptor_pdb, pose, folder, fragments)
    clash, void, mismatch = _geometry_signals(receptor_pdb, pose)
    medchem, data = _medicinal_signals(original, score)
    for fragment in fragments:
        fragment['interactions'] = [r for r in interactions if fragment['fragment_name'] in r['fragments']]
    return {'interaction_text': json.dumps(interactions, indent=2) if interactions else 'PLIP completed: no interactions detected.',
            'clash_text': clash, 'void_text': void, 'mismatch_text': mismatch,
            'frag_inter_match': json.dumps(fragments, indent=2), 'medchem_text': medchem,
            'mc_data': data, 'fragments': fragments}
