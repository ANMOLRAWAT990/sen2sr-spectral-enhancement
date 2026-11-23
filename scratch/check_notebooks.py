import json

paths = [
    'notebooks/04_finetuning_experiments/Train_SEN2SR_Lite_Compound_Loss.ipynb',
    'notebooks/04_finetuning_experiments/SEN2SR_Baseline_vs_Finetuned_Comparison.ipynb',
    'notebooks/06_full_model_experiments/Train_SEN2SR_Full_Compound_Loss.ipynb',
    'notebooks/06_full_model_experiments/SEN2SR_Full_Baseline_vs_Finetuned_Comparison.ipynb'
]

for p in paths:
    nb = json.load(open(p, encoding='utf-8'))
    print(f"\n=== {p}: {len(nb['cells'])} cells ===")
    for i, c in enumerate(nb['cells']):
        ct = c['cell_type']
        fl = c['source'][0].strip() if c['source'] else ''
        print(f"  [{i:02d}] {ct:8s}: {fl[:70]}")
