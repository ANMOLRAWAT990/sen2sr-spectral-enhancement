import json, ast, glob, os

def check_all_notebooks(dir_path):
    print("=" * 70)
    print("Validating notebooks in:", dir_path)
    print("=" * 70)
    nb_files = glob.glob(os.path.join(dir_path, '**/*.ipynb'), recursive=True)
    assert len(nb_files) > 0, f"No notebooks found in {dir_path}"
    errors = 0
    for p in sorted(nb_files):
        with open(p, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        cell_count = len(nb.get('cells', []))
        for idx, c in enumerate(nb.get('cells', [])):
            if c.get('cell_type') == 'code':
                code_str = "".join(c.get('source', []))
                clean_lines = [l for l in code_str.split('\n') if not l.strip().startswith(('%pip', '!', '%'))]
                clean_code = '\n'.join(clean_lines)
                try:
                    ast.parse(clean_code)
                except SyntaxError as e:
                    print(f"ERROR in {p} cell {idx}: {e}")
                    errors += 1
        rel_p = os.path.relpath(p, dir_path)
        print(f"  [OK] {rel_p} ({cell_count} cells)")
    print(f"\nCompleted {len(nb_files)} notebooks. Total syntax errors: {errors}\n")
    assert errors == 0, f"{errors} syntax errors detected!"

if __name__ == '__main__':
    check_all_notebooks('notebooks')
    if os.path.exists('G:/My Drive/Project_Folder/Notebooks'):
        check_all_notebooks('G:/My Drive/Project_Folder/Notebooks')
    print("ALL NOTEBOOKS PASSED AST SYNTAX VALIDATION.")
