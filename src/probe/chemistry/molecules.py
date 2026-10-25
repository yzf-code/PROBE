"""Canonical molecular identifiers, map labels, and shared property scoring."""

from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit.Contrib.SA_Score import sascorer


def molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        raise ValueError("A valid, nonempty molecular SMILES is required")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Use a single connected ligand, without salts or counterions")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(molecule(smiles), isomericSmiles=True)


def map_smiles(smiles: str) -> tuple[str, str]:
    """Give every atom a positive map label, independent of SMILES ordering."""
    mol = molecule(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    legend = []
    for atom in mol.GetAtoms():
        neighbors = ", ".join(str(n.GetAtomMapNum()) for n in atom.GetNeighbors())
        legend.append(
            f"{atom.GetAtomMapNum()}: {atom.GetSymbol()}, "
            f"aromatic={atom.GetIsAromatic()}, ring={atom.IsInRing()}, "
            f"available_H={atom.GetTotalNumHs()}, charge={atom.GetFormalCharge()}, "
            f"neighbors=[{neighbors}]"
        )
    return Chem.MolToSmiles(mol, isomericSmiles=True), "\n".join(legend)


def chemical_metrics(smiles: str) -> dict[str, float]:
    mol = molecule(smiles)
    raw_sa = sascorer.calculateScore(mol)
    sa = round((10.0 - raw_sa) / 9.0, 2)
    return {"qed": float(QED.qed(mol)), "sa": sa, "logp": float(Crippen.MolLogP(mol))}
