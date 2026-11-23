with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(1510, 1579):
    print(f"{i+1:4d}: {lines[i]}", end='')

