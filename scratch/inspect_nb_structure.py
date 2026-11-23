import json

def inspect_nb(path):
    print('='*50, path, '='*50)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print('Total cells:', len(nb['cells']))
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        first_line = src.strip().split('\n')[0] if src.strip() else '(empty)'
        print(f"Cell {i:2d} [{c['cell_type']}]: {first_line[:80]}")

inspect_nb('notebooks/04_finetuning_experiments/Train_SEN2SR_Lite_Compound_Loss.ipynb')
inspect_nb('notebooks/04_finetuning_experiments/SEN2SR_Baseline_vs_Finetuned_Comparison.ipynb')
inspect_nb('notebooks/06_full_model_experiments/Train_SEN2SR_Full_Compound_Loss.ipynb')
inspect_nb('notebooks/06_full_model_experiments/SEN2SR_Full_Baseline_vs_Finetuned_Comparison.ipynb')
