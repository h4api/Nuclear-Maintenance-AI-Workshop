"""Produce a local HTML report from two actual evaluation runs. Standard library only."""
import argparse
import html
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('trained', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    before, after = [json.loads(x.read_text()) for x in (a.baseline, a.trained)]
    if (before['data_sha256'], before['model_config_sha256']) != (after['data_sha256'], after['model_config_sha256']):
        p.error('Reports must use the same dataset and base model configuration.')
    if a.out.exists():
        p.error('Output exists; choose a new name.')
    esc = html.escape
    rows = ''.join(f'<tr><td>{esc(k)}</td><td>{before["totals"][k]}/{before["n"]}</td>'
                   f'<td>{after["totals"][k]}/{after["n"]}</td></tr>' for k in before['totals'])
    cards = []
    bmap = {r['id']: r for r in before['records']}
    for r in after['records']:
        b = bmap[r['id']]
        cards.append(f'<details><summary>{esc(r["id"])} · {esc(r["case"])} · '
            f'exact match: {r["scores"]["exact_match"]}</summary><h3>Input</h3><pre>{esc(r["input"])}</pre>'
            f'<h3>Expected</h3><pre>{esc(json.dumps(r["target"], indent=2))}</pre>'
            f'<h3>Baseline</h3><pre>{esc(b["prediction"])}</pre>'
            f'<h3>Adapter</h3><pre>{esc(r["prediction"])}</pre></details>')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Actual evaluation results</title><style>body{font:17px/1.6 system-ui;background:#edf3f8;color:#132b40;max-width:1000px;margin:40px auto;padding:24px}'
        'table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #bdcbd6}'
        'details{background:white;padding:20px;margin:16px 0}summary{cursor:pointer;font-weight:700}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:40px}</style>'
        f'<h1>Baseline → adapter</h1><p>Actual {esc(after["split"])} results · {after["n"]} fictional records.</p>'
        '<p>Higher is better except unsupported_department (lower is better). Invalid schemas count as wrong for every field. '
        'This narrow diagnostic does not measure all unsupported claims. Expand records for human review.</p>'
        '<table><tr><th>Metric</th><th>Baseline</th><th>Adapter</th></tr>'+rows+'</table>'
        '<h2>Inspect every prediction</h2>'+''.join(cards)+'</html>')
    print('Report:', a.out)

if __name__ == '__main__':
    main()
