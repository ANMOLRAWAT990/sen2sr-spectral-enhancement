import json

nb_local = json.load(open('notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb', encoding='utf-8'))
nb_gdrive = json.load(open('G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb', encoding='utf-8'))

print("Local cells:", len(nb_local['cells']))
print("GDrive cells:", len(nb_gdrive['cells']))

all_match = True
for idx, (c1, c2) in enumerate(zip(nb_local['cells'], nb_gdrive['cells'])):
    src1 = ''.join(c1['source'])
    src2 = ''.join(c2['source'])
    if src1 != src2:
        print(f"Difference in cell {idx} ({c1.get('id')}):")
        print("  Local:", src1[:60])
        print("  GDrive:", src2[:60])
        all_match = False

if all_match:
    print("ALL CELLS MATCH 100% IN CODE SOURCE!")

