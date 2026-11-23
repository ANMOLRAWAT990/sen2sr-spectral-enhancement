import json

with open(r'G:\My Drive\Project_Folder\Notebooks\07_cascaded_1.5m_vhr_experiments\SEN2SR_1.5Meter_MultiStage_Comparison.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell.get('id') == 'cell11_code' or idx == 23:
        print(f"Cell {idx} ({cell.get('id')}):")
        print(''.join(cell['source']))

