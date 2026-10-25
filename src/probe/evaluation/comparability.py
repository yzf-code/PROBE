"""Check input and protocol compatibility before ranking evaluations."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from collections.abc import Mapping
from numbers import Real
from probe.chemistry.molecules import canonical_smiles


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _input_hash(inputs, field, identity, warnings):
    saved = inputs.get('sha256', {})
    if isinstance(saved, Mapping) and field in saved:
        value = saved[field]
        if isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value):
            return value.lower()
        warnings.append(f'{identity}: recorded sha256.{field} is malformed')
        return None
    path = inputs.get(field)
    if not isinstance(path, (str, Path)) or not str(path):
        warnings.append(f'{identity}: no recorded hash or file path for {field}')
        return None
    try:
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        digest = hashlib.sha256()
        with file_path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        warnings.append(f'{identity}: no recorded {field} hash and file cannot be read ({type(exc).__name__})')
        return None


def _version(package, dependencies, *, current, warnings):
    if current:
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            value = None
    else:
        value = dependencies.get(package)
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {'unknown', 'n/a', 'none'}:
        warnings.append(f'Missing {"runtime" if current else "recorded docking"} {package} version')
        return None
    return value.strip()


def _current_file_matches(inputs, field, expected, identity, warnings):
    # Re-docking consumes the current files. Historical hashes must not hide a
    # changed receptor/reference after a run folder has been copied or edited.
    actual = _input_hash({field: inputs.get(field)}, field, identity, warnings)
    if actual is None or expected is None:
        return False
    if actual != expected:
        warnings.append(f'{identity}: current {field} content differs from the recorded input hash')
        return False
    return True


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _docking_protocol(row, identity, warnings):
    protocol = row.get('docking_protocol')
    if not isinstance(protocol, Mapping):
        warnings.append(f'{identity}: missing per-molecule docking_protocol')
        return None
    parsed = {}
    for field in ('seed', 'cpu', 'exhaustiveness'):
        value = protocol.get(field)
        if type(value) is not int or value < 1 or (field == 'seed' and value >= 2**31):
            warnings.append(f'{identity}: docking_protocol.{field} must be a positive integer')
            return None
        parsed[field] = value
    for field in ('center', 'box_size'):
        vector = protocol.get(field)
        if (not isinstance(vector, (list, tuple)) or len(vector) != 3 or
                not all(_finite(v) for v in vector) or
                (field == 'box_size' and not all(v > 0 for v in vector))):
            warnings.append(f'{identity}: docking_protocol.{field} must contain three finite valid coordinates')
            return None
        parsed[field] = [float(v) for v in vector]
    return parsed


def comparison_metadata(records, rows, manifest, *, redock, exhaustiveness, seed, cpu, mrr_mode):
    """Return cohort/protocol fingerprints and completeness flags for ranking."""
    if mrr_mode not in {'legacy', 'corrected'}:
        raise ValueError('mrr_mode must be legacy or corrected')
    if not isinstance(manifest, Mapping):
        raise ValueError('manifest must be an object')
    if not isinstance(records, list) or not isinstance(rows, list):
        raise ValueError('records and rows must be lists')
    warnings = []
    dependencies = manifest.get('dependencies', {})
    if not isinstance(dependencies, Mapping): dependencies = {}
    ids = [record.get('id') if isinstance(record, Mapping) else None for record in records]
    if (not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError('records must have unique nonempty IDs')
    row_ids = [row.get('id') if isinstance(row, Mapping) else None for row in rows]
    if any(not isinstance(i, str) or not i for i in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError('rows must have unique nonempty IDs')
    unexpected = set(row_ids)-set(ids)
    if unexpected:
        raise ValueError(f'Evaluation rows contain IDs outside the requested cohort: {sorted(unexpected)}')
    by_id = dict(zip(row_ids, rows))
    cohort, per_case = [], []
    cohort_complete, protocol_complete = True, True
    for record in sorted(records, key=lambda r: r['id']):
        identity = record['id']
        inputs = record.get('input')
        if not isinstance(inputs, Mapping): inputs = {}
        try:
            original = canonical_smiles(inputs.get('smiles'))
        except (ValueError, TypeError, RuntimeError):
            original = None
            warnings.append(f'{identity}: original input SMILES are missing or invalid')
        receptor_hash = _input_hash(inputs, 'receptor', identity, warnings)
        reference_hash = _input_hash(inputs, 'reference_ligand', identity, warnings)
        if not all((original, receptor_hash, reference_hash)):
            cohort_complete = False
        if redock:
            matches = [_current_file_matches(inputs, field, expected, identity, warnings)
                       for field, expected in (('receptor', receptor_hash),
                                               ('reference_ligand', reference_hash))]
            if not all(matches):
                cohort_complete = False
                protocol_complete = False
        cohort.append({'id': identity, 'original_smiles': original,
                       'receptor_sha256': receptor_hash, 'reference_ligand_sha256': reference_hash})
        prepared_hash = None
        saved_hashes = inputs.get('sha256')
        provided_pdbqt = bool(inputs.get('receptor_pdbqt')) or (
            isinstance(saved_hashes, Mapping) and 'receptor_pdbqt' in saved_hashes)
        if provided_pdbqt:
            prepared_hash = _input_hash(inputs, 'receptor_pdbqt', identity, warnings)
            if not prepared_hash: protocol_complete = False
            if redock and not _current_file_matches(inputs, 'receptor_pdbqt', prepared_hash, identity, warnings):
                protocol_complete = False
        row = by_id.get(identity)
        if row is None or row.get('status') != 'ok':
            protocol_complete = False
            warnings.append(f'{identity}: ranking requires a completed evaluation row for every requested input')
        parsed = _docking_protocol(row or {}, identity, warnings)
        if parsed is None:
            protocol_complete = False
        elif redock and any(parsed[k] != v for k, v in (('seed', seed), ('cpu', cpu), ('exhaustiveness', exhaustiveness))):
            protocol_complete = False
            warnings.append(f'{identity}: observed docking settings differ from the requested re-docking settings')
        if row is not None:
            if row.get('mrr_mode') != mrr_mode:
                protocol_complete = False
                warnings.append(f'{identity}: missing or inconsistent per-row MRR mode')
            if not all(_finite(row.get(k)) for k in ('vina_score', 'qed', 'sa')):
                protocol_complete = False
                warnings.append(f'{identity}: missing finite core measurements')
        per_case.append({'id': identity, 'docking': parsed,
                         'receptor_preparation': 'provided_pdbqt' if provided_pdbqt else 'automatic_meeko',
                         'receptor_pdbqt_sha256': prepared_hash})
    property_rdkit = _version('rdkit', dependencies, current=True, warnings=warnings)
    docking_versions = {name: _version(name, dependencies, current=redock, warnings=warnings)
                        for name in ('rdkit', 'vina', 'meeko')}
    if property_rdkit is None or any(v is None for v in docking_versions.values()):
        protocol_complete = False
    protocol = {
        'comparison_schema': 1, 'property_rdkit_version': property_rdkit,
        'docking_versions': docking_versions, 'mrr_mode': mrr_mode,
        'sa_definition': 'round((10-raw_rdkit_sa)/9,2)',
        'success_definition': 'vina<-8.18 and qed>0.25 and sa>0.59',
        'pass_ratio_denominator': 'all_requested_inputs', 'cases': per_case,
    }
    return {'cohort_id': _fingerprint(cohort) if cohort_complete else None,
            'cohort_complete': cohort_complete,
            'protocol_id': _fingerprint(protocol) if protocol_complete else None,
            'protocol_complete': protocol_complete, 'warnings': warnings,
            'cohort': cohort, 'protocol': protocol}
