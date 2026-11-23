import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

train_path = r'G:\My Drive\Project_Folder\Notebooks\06_full_model_experiments\Train_SEN2SR_Full_Compound_Loss.ipynb'
print(f"=== CHECKING TRAIN NOTEBOOK: {train_path} ===")
nb_train = json.load(open(train_path, encoding='utf-8'))
for idx, c in enumerate(nb_train['cells']):
    if c['cell_type'] == 'code':
        ec = c.get('execution_count')
        print(f"\n--- Cell {idx} (count={ec}) ---")
        for out in c.get('outputs', []):
            if 'text' in out:
                print(''.join(out['text']).strip()[:500])
            elif 'data' in out:
                print("Data keys:", list(out['data'].keys()))
                if 'text/plain' in out['data']:
                    print(''.join(out['data']['text/plain']).strip()[:200])

