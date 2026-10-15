"""Create or verify a relative SHA256 inventory for workshop transfer. No dependencies."""
import argparse
import json
from common import ROOT, digest

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['create', 'verify'])
a = p.parse_args()
manifest = ROOT / 'transfer-sha256.json'
excluded = {'.venv', '__pycache__', '.cache', '.git', 'runs'}
if a.mode == 'create':
    files = {str(x.relative_to(ROOT)): digest(x) for x in ROOT.rglob('*')
             if x.is_file() and x != manifest
             and not any(part in excluded for part in x.relative_to(ROOT).parts)}
    manifest.write_text(json.dumps(files, indent=2, sort_keys=True))
    print('Inventoried', len(files), 'files; virtual environments, caches and runs excluded.')
else:
    files = json.loads(manifest.read_text())
    failures = []
    for relative, expected in files.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            failures.append(relative)
    if failures:
        raise SystemExit('Verification FAILED: ' + ', '.join(failures))
    print('PASS:', len(files), 'listed files match. Unlisted files are not checked.')
