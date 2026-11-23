import json, base64

nb = json.load(open('G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb', encoding='utf-8'))

for idx, cell in enumerate(nb['cells']):
    if cell.get('id') == 'cell11_code' or idx == 22:
        for out in cell.get('outputs', []):
            if 'image/png' in out.get('data', {}):
                img_data = out['data']['image/png']
                with open('scratch/cell11_output_from_gdrive_nb.png', 'wb') as f:
                    f.write(base64.b64decode(img_data))
                print('Saved cell11 image!')
                break

