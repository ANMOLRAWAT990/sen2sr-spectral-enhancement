import os, glob, datetime, json

results_dir = 'G:/My Drive/Project_Folder/Results'
files = glob.glob(os.path.join(results_dir, '*'))
print(f'Total files in {results_dir}: {len(files)}')
for f in sorted(files, key=os.path.getmtime, reverse=True)[:15]:
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    size = os.path.getsize(f)
    print(f'{mtime.strftime("%Y-%m-%d %H:%M:%S")} | {size:>10,} bytes | {os.path.basename(f)}')

print('\n--- Checking Notebook Executions in Drive ---')
for nb_path in [
    'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb',
    'G:/My Drive/Project_Folder/Notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb'
]:
    if os.path.exists(nb_path):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(nb_path))
        print(f'\nNotebook: {os.path.basename(nb_path)} (Modified: {mtime.strftime("%Y-%m-%d %H:%M:%S")})')
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] == 'code':
                exec_count = c.get('execution_count')
                outputs = c.get('outputs', [])
                out_summary = []
                for o in outputs:
                    if o.get('output_type') == 'stream':
                        text = ''.join(o.get('text', [])).strip()
                        lines = text.splitlines()
                        summary_text = lines[-1] if lines else ''
                        out_summary.append(f'stream: {summary_text[:80]}')
                    elif o.get('output_type') in ['display_data', 'execute_result']:
                        data = o.get('data', {})
                        keys = list(data.keys())
                        out_summary.append(f'data({",".join(keys)})')
                    elif o.get('output_type') == 'error':
                        ename = o.get('ename')
                        evalue = o.get('evalue')
                        out_summary.append(f'ERROR: {ename}: {evalue}')
                if exec_count or outputs:
                    print(f'  Cell {i} [exec {exec_count}]: {" | ".join(out_summary)[:100]}')

