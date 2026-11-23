import json

nb = json.load(open('notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb', encoding='utf-8'))

for cell in nb['cells']:
    if cell.get('id') == 'cell7_code':
        print(''.join(cell['source']))

