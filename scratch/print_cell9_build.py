with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('def apply_universal_vhr_crisp')
start = max(0, idx - 1500)
end = min(len(text), idx + 2500)
print(text[start:end])

