with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in build script: {len(lines)}")
for idx, line in enumerate(lines):
    if 'cell9_code' in line or 'cell11_code' in line:
        print(f"Line {idx+1}: {line.strip()}")

