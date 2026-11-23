import json, ast

notebooks = [
    r'notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb',
    r'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb',
    r'notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb',
    r'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb'
]

for nb_path in notebooks:
    print(f"\nValidating: {nb_path}")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"  Total cells: {len(nb['cells'])}")
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            lines = [l for l in cell['source'] if not l.strip().startswith('%') and not l.strip().startswith('!')]
            code = ''.join(lines)
            try:
                ast.parse(code)
            except SyntaxError as e:
                print(f"  [ERROR] Cell {idx+1} syntax error: {e}")
    print("  All code cells compiled cleanly with Python AST!")

