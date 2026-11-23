import os, json

with open('G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb', 'r', encoding='utf-8') as f:
    nb_train = json.load(f)

print('=== Train Notebook Cell 16 Outputs ===')
cell16 = nb_train['cells'][16]
for o in cell16.get('outputs', []):
    if o.get('output_type') == 'stream':
        print(''.join(o.get('text', [])))

with open('G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb', 'r', encoding='utf-8') as f:
    nb_comp = json.load(f)

print('=== Comparison Notebook Cell 22 Outputs (Ground Truth Showcase) ===')
cell22 = nb_comp['cells'][22]
for o in cell22.get('outputs', []):
    if o.get('output_type') == 'stream':
        print(''.join(o.get('text', [])))

print('=== Comparison Notebook Cell 20 Outputs (Benchmark Table) ===')
cell20 = nb_comp['cells'][20]
for o in cell20.get('outputs', []):
    if o.get('output_type') == 'stream':
        print(''.join(o.get('text', [])))

