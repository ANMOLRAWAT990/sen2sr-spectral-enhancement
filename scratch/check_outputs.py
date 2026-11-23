import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'G:\My Drive\Project_Folder\Notebooks\06_full_model_experiments\SEN2SR_Full_Baseline_vs_Finetuned_Comparison.ipynb'
if not os.path.exists(nb_path):
    print("Not found:", nb_path)
    exit(1)

nb = json.load(open(nb_path, encoding='utf-8'))
for idx, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        ec = c.get('execution_count')
        print(f"\n==================== CELL {idx} (count={ec}) ====================")
        for out in c.get('outputs', []):
            if 'text' in out:
                print(''.join(out['text']).strip())
            elif 'data' in out:
                keys = list(out['data'].keys())
                print(f"[DATA OUTPUT: {keys}]")
                if 'text/plain' in out['data']:
                    print(''.join(out['data']['text/plain']).strip())
