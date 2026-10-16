"""Persist numerical runs with source fingerprints."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import uuid
import math
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def source_manifest():
    files = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'config').glob('*.json'))
    files += sorted(ROOT.glob('run_*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    files += sorted((ROOT / 'prompt').glob('*.txt'))
    files += [ROOT / 'main.py', ROOT / 'requirements.txt', ROOT / 'requirements-ppo-lstm.txt']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'git_commit': git('rev-parse', 'HEAD'),
            'tracked_changes': git('status', '--short', '--untracked-files=no'),
            'source_sha256': hashes}


def save_run(result, args, *, root=None):
    """NPZ contains numeric arrays only; JSON never includes environment/API keys."""
    parent = Path(root or getattr(args, 'output_dir', 'output')) / 'raw_runs'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:10]
    directory = parent / run_id
    directory.mkdir(parents=True, exist_ok=False)
    arrays, metadata = {}, {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif key != 'archive_path':
            metadata[key] = value
    np.savez_compressed(directory / 'trace.npz', **arrays)
    metadata['source'] = source_manifest()
    with zipfile.ZipFile(directory / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for relative_path in metadata['source']['source_sha256']:
            bundle.write(ROOT / relative_path, arcname=relative_path)
    metadata['created_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['inference_time_model'] = result.get('inference_time_model', 'not_applicable')
    metadata['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                             if k in {'case_number', 'sim_time', 'dt', 'seed', 'llm', 'llm_provider',
                                      'llm_trigger_mode', 'llm_fixed_interval', 'llm_high_level_update_steps',
                                      'disable_memory', 'disable_rule_validator', 'llm_risk_threshold',
                                      'rule_baseline', 'low_level_planner', 'method', 'ppo_checkpoint',
                                      'llm_latency_mode', 'llm_latency_s', 'llm_latency_jitter_s',
                                      'llm_timeout_s', 'llm_max_decision_age_s', 'llm_failure_probability',
                                      'llm_retry_interval_s', 'vo_prediction_model', 'vo_response_dt_s'}}
    (directory / 'run.json').write_text(json.dumps(_json_value(metadata), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return str(directory.resolve())


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    with np.load(directory / 'trace.npz', allow_pickle=False) as arrays:
        return {**metadata, **{key: arrays[key].copy() for key in arrays.files}}
