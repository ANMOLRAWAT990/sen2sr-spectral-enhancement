with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Find cell9 definition
c9_start = text.find('cell_id="cell9_code"')
if c9_start != -1:
    print("Found cell9_code at", c9_start)
    block_start = text.rfind('cells.append(make_cell("code",', 0, c9_start)
    print("Block start:", block_start)
    print("Cell 9 source preview:\n", text[block_start:block_start+1500])

c11_start = text.find('cell_id="cell11_code"')
if c11_start != -1:
    print("\nFound cell11_code at", c11_start)
    block_start = text.rfind('cells.append(make_cell("code",', 0, c11_start)
    print("Block start:", block_start)
    print("Cell 11 source preview:\n", text[block_start:block_start+1500])

