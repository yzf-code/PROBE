"""Open-source Vina evaluation with isolated, persistent per-run work files."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from .molecules import chemical_metrics


def reference_box(reference_ligand):
    path = Path(reference_ligand)
    if path.suffix.lower() == '.pdb':
        ref = Chem.MolFromPDBFile(str(path), removeHs=False)
    else:
        ref = Chem.MolFromMolFile(str(path), removeHs=False)
    if ref is None or not ref.GetNumConformers():
        raise ValueError(f'Cannot read a 3D reference ligand from {path}')
    ref = Chem.AddHs(ref, addCoords=True)
    positions = ref.GetConformer().GetPositions()
    if not np.isfinite(positions).all():
        raise ValueError('Reference ligand has nonfinite coordinates')
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    return ((upper+lower)/2).tolist(), (upper-lower+12.0).tolist()


class VinaEvaluator:
    """One receptor/reference box and one scoring protocol for an optimization run."""
    def __init__(self, receptor_pdb, reference_ligand, work_dir, exhaustiveness=16,
                 seed=42, cpu=1, receptor_pdbqt=None):
        self.receptor_pdb = Path(receptor_pdb).expanduser().resolve()
        self.reference_ligand = Path(reference_ligand).expanduser().resolve()
        for path in (self.receptor_pdb, self.reference_ligand):
            if not path.is_file(): raise FileNotFoundError(path)
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if exhaustiveness < 1 or cpu < 1 or not 0 < seed < 2**31:
            raise ValueError('exhaustiveness/cpu must be positive; seed must be in [1, 2^31)')
        self.exhaustiveness, self.seed, self.cpu = exhaustiveness, seed, cpu
        self.center, self.box_size = reference_box(self.reference_ligand)
        self.receptor_pdbqt = Path(receptor_pdbqt).expanduser().resolve() if receptor_pdbqt else None
        if self.receptor_pdbqt and not self.receptor_pdbqt.is_file():
            raise FileNotFoundError(self.receptor_pdbqt)
        self._cache = {}

    def _prepare_receptor(self):
        if self.receptor_pdbqt is not None:
            return self.receptor_pdbqt
        command = shutil.which('mk_prepare_receptor.py')
        if command is None:
            raise RuntimeError('Install Meeko >=0.6 or set the CSV receptor_pdbqt column')
        # Do not write beside the user's dataset or reuse a stale receptor file.
        folder = Path(tempfile.mkdtemp(prefix='receptor-', dir=self.work_dir))
        prepared = folder / 'receptor.pdbqt'
        proc = subprocess.run(
            [command, '--read_pdb', str(self.receptor_pdb), '--write_pdbqt', str(prepared)],
            cwd=folder, capture_output=True, text=True, timeout=300, check=False,
        )
        (folder / 'preparation.log').write_text(proc.stdout+'\n'+proc.stderr, encoding='utf-8')
        if proc.returncode or not prepared.is_file():
            raise RuntimeError(f'Receptor preparation failed; see {folder / "preparation.log"}. '
                               'Prepare the receptor explicitly or set the CSV receptor_pdbqt column.')
        self.receptor_pdbqt = prepared
        return prepared

    def evaluate(self, smiles):
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
            from vina import Vina
        except ImportError as exc:
            raise RuntimeError('Docking requires modern Meeko >=0.6 and AutoDock Vina >=1.2.5') from exc
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError('Docking requires one valid connected SMILES molecule')
        for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical in self._cache:
            return dict(self._cache[canonical])
        metrics = chemical_metrics(canonical)
        receptor = self._prepare_receptor()
        run = Path(tempfile.mkdtemp(prefix=hashlib.sha256(canonical.encode()).hexdigest()[:12]+'-', dir=self.work_dir))
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        conformers = list(AllChem.EmbedMultipleConfs(mol, numConfs=3, params=params))
        if not conformers:
            raise RuntimeError('RDKit could not generate a 3D conformer')
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=self.cpu)
        else:
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=self.cpu)
        prepared_mol = Chem.Mol(mol)
        first = Chem.Conformer(mol.GetConformer(conformers[0]))
        prepared_mol.RemoveAllConformers()
        prepared_mol.AddConformer(first, assignId=True)
        setups = MoleculePreparation().prepare(prepared_mol)
        if len(setups) != 1:
            raise RuntimeError('Ligand preparation produced multiple setups; choose an explicit ligand state')
        pdbqt, ok, error = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise RuntimeError(f'Meeko ligand preparation failed: {error}')
        (run/'input.pdbqt').write_text(pdbqt, encoding='utf-8')
        engine = Vina(sf_name='vina', cpu=self.cpu, seed=self.seed, verbosity=0)
        engine.set_receptor(str(receptor))
        engine.set_ligand_from_string(pdbqt)
        engine.compute_vina_maps(center=self.center, box_size=self.box_size)
        engine.dock(exhaustiveness=self.exhaustiveness, n_poses=1)
        energies = engine.energies(n_poses=1)
        if len(energies) != 1 or not math.isfinite(float(energies[0][0])):
            raise RuntimeError('Vina did not return a finite docking score')
        docked = engine.poses(n_poses=1)
        (run/'docked.pdbqt').write_text(docked, encoding='utf-8')
        molecules = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(docked, skip_typing=True))
        if len(molecules) != 1 or molecules[0] is None:
            raise RuntimeError('Cannot recover a chemically typed docked pose from Meeko metadata')
        pose = molecules[0]
        if not pose.GetNumConformers():
            raise RuntimeError('Docked pose has no coordinates')
        pose_path = run/'docked.sdf'
        with Chem.SDWriter(str(pose_path)) as writer:
            writer.write(pose, confId=0)
        result = {**metrics, 'vina_score': float(energies[0][0]), 'pose_path': str(pose_path),
                  'docking_protocol': {'seed': self.seed, 'cpu': self.cpu,
                                       'exhaustiveness': self.exhaustiveness,
                                       'center': self.center, 'box_size': self.box_size}}
        self._cache[canonical] = result
        return dict(result)
