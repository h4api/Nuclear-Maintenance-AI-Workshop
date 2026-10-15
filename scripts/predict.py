"""Classify one local text file with a saved adapter. No Ollama server needed."""
import argparse
from pathlib import Path
from common import ROOT, generate, load_model

p = argparse.ArgumentParser()
p.add_argument('record', type=Path)
p.add_argument('--adapter', type=Path, default=ROOT / 'runs/run01/adapter')
p.add_argument('--base-only', action='store_true', help='Generate with the original model, without an adapter.')
a = p.parse_args()
text = a.record.read_text()
if not text.strip():
    p.error('Record is empty')
if len(text) > 4000:
    p.error('Workshop input limit is 4000 characters; use a short administrative record.')
tokenizer, model = load_model(ROOT / 'models/qwen-base', None if a.base_only else a.adapter)
print(generate(tokenizer, model, text))
