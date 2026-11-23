import os, json, pandas as pd

results_dir = 'G:/My Drive/Project_Folder/Results'

# 1. Edge Acuity CSV
csv_path = os.path.join(results_dir, 'SEN2SR_1.5M_Edge_Acuity_Improvement.csv')
if os.path.exists(csv_path):
    print('=== SEN2SR_1.5M_Edge_Acuity_Improvement.csv ===')
    df = pd.read_csv(csv_path)
    print(df.to_string(index=False))

# 2. Extract detailed outputs from Comparison Notebook
comp_nb_path = 'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb'
if os.path.exists(comp_nb_path):
    with open(comp_nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print('\n=== Detailed Outputs from SEN2SR_1.5Meter_MultiStage_Comparison.ipynb ===')
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream':
                    text = ''.join(o.get('text', [])).strip()
                    for line in text.splitlines():
                        if any(k in line for k in ['PSNR', 'Benchmark', 'Acuity', 'Sobel', 'Laplacian', 'Drift', 'Gain', 'dB', 'Shape', 'completed in', 'finished in', 'Range']):
                            print(f'  Cell {i}: {line}')

# 3. Extract detailed outputs from Train Notebook
train_nb_path = 'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb'
if os.path.exists(train_nb_path):
    with open(train_nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print('\n=== Detailed Outputs from Train_SEN2SR_Cascaded_1.5m_VHR.ipynb ===')
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream':
                    text = ''.join(o.get('text', [])).strip()
                    for line in text.splitlines():
                        if any(k in line for k in ['Epoch', 'Train:', 'Val:', 'PSNR', 'Deviation', 'Radiometry', 'finished in']):
                            print(f'  Cell {i}: {line}')

