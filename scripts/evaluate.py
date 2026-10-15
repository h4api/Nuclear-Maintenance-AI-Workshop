"""Fresh-process baseline or saved-adapter evaluation; no fitting or repairs to outputs."""
import argparse
import json
from pathlib import Path
from common import ROOT, digest, generate, load_model, read_rows, score

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path, default=ROOT / 'models/qwen-base')
    args = p.parse_args()
    if args.out.exists():
        p.error('Output already exists. Choose a new output name.')
    tokenizer, model = load_model(args.model, args.adapter)
    data = ROOT / 'data' / (args.split + '.jsonl')
    records = []
    for row in read_rows(data):
        raw = generate(tokenizer, model, row['input'])
        records.append({'id': row['id'], 'case': row['case'], 'input': row['input'],
            'target': row['target'], 'prediction': raw, 'scores': score(raw, row['target'])})
        print(row['id'], records[-1]['scores']['exact_match'], flush=True)
    totals = {k: sum(r['scores'][k] for r in records) for k in records[0]['scores']}
    report = {'split': args.split, 'n': len(records), 'data_sha256': digest(data),
        'adapter': str(args.adapter) if args.adapter else None,
        'model_config_sha256': digest(args.model / 'config.json'),
        'totals': totals, 'rates': {k: v/len(records) for k, v in totals.items()},
        'records': records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({'n': report['n'], 'totals': totals}, indent=2))
    print('Saved', args.out)

if __name__ == '__main__':
    main()
