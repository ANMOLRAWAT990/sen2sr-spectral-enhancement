with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(1260, 1375):
    print(f"{i+1:4d}: {lines[i]}", end='')

