"""Train only on train.jsonl. Evaluate development data with evaluate.py separately."""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT, SYSTEM, digest, load_model, prompt_messages, read_rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, default=ROOT / 'runs' / 'run01')
    p.add_argument('--model', type=Path, default=ROOT / 'models' / 'qwen-base')
    args = p.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        p.error('epochs and lr must be positive')
    if args.out.exists():
        p.error('Output exists. Use a new --out path; previous results are preserved.')
    import torch
    from peft import LoraConfig, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, base = load_model(args.model)
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=8,
        lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'], bias='none'))
    model.config.use_cache = False
    model.print_trainable_parameters()
    rows = read_rows(ROOT / 'data' / 'train.jsonl')
    batches = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(prompt_messages(row['input']),
            tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(prompt_messages(row['input']) + [
            {'role': 'assistant', 'content': json.dumps(row['target'], separators=(',', ':'))}],
            tokenize=True, add_generation_prompt=False)
        if full[:len(prompt)] != prompt:
            raise ValueError('Chat template prefix mismatch')
        if len(full) > 768:
            raise ValueError(f"Record {row['id']} exceeds 768 tokens; do not silently truncate labels.")
        ids = torch.tensor([full])  # CPU until each step; no padded batch.
        labels = ids.clone()
        labels[:, :len(prompt)] = -100  # Score assistant answer tokens only.
        batches.append((row['id'], ids, labels))
    args.out.mkdir(parents=True)
    metadata = {'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
        'train_sha256': digest(ROOT / 'data/train.jsonl'),
        'system_prompt_sha256': __import__('hashlib').sha256(SYSTEM.encode()).hexdigest(),
        'model_config_sha256': digest(args.model / 'config.json'),
        'model_revision': json.loads((ROOT / 'models/model-lock.json').read_text())
            if (ROOT / 'models/model-lock.json').exists() else 'unrecorded',
        'gpu': torch.cuda.get_device_name(0), 'train_examples': len(rows),
        'lora': {'r': 8, 'alpha': 16, 'targets': ['q_proj', 'v_proj']}}
    (args.out / 'run.json').write_text(json.dumps(metadata, indent=2))
    (args.out / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    step, start = 0, time.monotonic()
    with (args.out / 'loss.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            random.shuffle(batches)
            losses = []
            for record_id, cpu_ids, cpu_labels in batches:
                ids, labels = cpu_ids.to('cuda'), cpu_labels.to('cuda')
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; training stopped.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                value = loss.item()
                losses.append(value)
                log.write(json.dumps({'step': step, 'epoch': epoch+1, 'id': record_id, 'loss': value})+'\n')
                if step % 10 == 0:
                    print(f'Step {step}/{len(rows)*args.epochs}: loss {value:.4f}', flush=True)
            print(f'Epoch {epoch+1}: mean training loss {sum(losses)/len(losses):.4f}')
    model.save_pretrained(args.out / 'adapter')
    tokenizer.save_pretrained(args.out / 'adapter')
    metadata.update(seconds=time.monotonic()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    (args.out / 'run.json').write_text(json.dumps(metadata, indent=2))
    print('Saved:', args.out, '\nNext: evaluate on dev; keep test sealed until configuration is fixed.')

if __name__ == '__main__':
    main()
