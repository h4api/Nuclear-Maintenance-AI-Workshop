"""CONNECTED staging step only. Never reads training data or uploads anything."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from common import ROOT

repo = 'Qwen/Qwen2.5-0.5B-Instruct'
destination = ROOT / 'models' / 'qwen-base'
lock = ROOT / 'models' / 'model-lock.json'
if lock.exists():
    revision = json.loads(lock.read_text())['revision']
else:
    revision = HfApi().model_info(repo).sha
snapshot_download(repo, revision=revision, local_dir=destination,
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.md', 'LICENSE*'])
lock.write_text(json.dumps({'repo': repo, 'revision': revision}, indent=2))
print('Prepared local model:', destination)
print('Pinned revision:', revision)
