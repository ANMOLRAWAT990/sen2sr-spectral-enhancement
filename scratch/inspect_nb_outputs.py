import json

nb = json.load(open('G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb', encoding='utf-8'))

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        cid = cell.get('id', f'cell_{idx}')
        exec_count = cell.get('execution_count')
        outputs = cell.get('outputs', [])
        print(f"\n--- Code Cell {idx} ({cid}), execution_count={exec_count} ---")
        for out in outputs:
            otype = out.get('output_type')
            if otype == 'stream':
                print(f"[{out.get('name')}] {''.join(out.get('text', []))[:300]}")
            elif otype in ('display_data', 'execute_result'):
                data_keys = list(out.get('data', {}).keys())
                print(f"[{otype}] keys: {data_keys}")
                if 'text/plain' in out.get('data', {}):
                    print(f"   text/plain: {out['data']['text/plain'][:200]}")
            elif otype == 'error':
                print(f"[ERROR] {out.get('ename')}: {out.get('evalue')}")

