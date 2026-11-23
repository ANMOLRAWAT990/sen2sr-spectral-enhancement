with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's locate the code string for cell9_code
start_needle = 'cells.append(make_cell("code", """def safe_percentile'
end_needle = 'print(f\'4-Way comparison matrix saved to: {comp4way_path}\')""", cell_id="cell9_code"))'

p_start = text.find(start_needle)
p_end = text.find(end_needle)

print("p_start:", p_start, "p_end:", p_end)
if p_start != -1 and p_end != -1:
    print("Found cell9_code block! Length:", p_end - p_start)

