import json, ast

for path in [
    'notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb',
    'notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb',
    'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb',
    'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb'
]:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f'Checking {path} ({len(nb["cells"])} cells)...')
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            clean_lines = []
            for line in code.splitlines():
                if line.strip().startswith('%') or line.strip().startswith('!'):
                    continue
                clean_lines.append(line)
            clean_code = '\n'.join(clean_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                print(f'  [FAIL] Cell {i}: {e}')
                raise
    print(f'  [PASS] All code cells valid in {path}!')
print('\n100% of notebooks passed syntax validation!')

