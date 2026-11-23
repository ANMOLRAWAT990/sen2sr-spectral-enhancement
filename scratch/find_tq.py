lines = open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8').readlines()
s = ''.join(lines[1253:1386])
print("Starts with:", repr(s[:50]))
print("Ends with:", repr(s[-50:]))
# Count triple quotes
tq_indices = []
pos = 0
while True:
    p = s.find('"""', pos)
    if p == -1: break
    tq_indices.append(p)
    pos = p + 3
print("Triple quote positions inside block:", tq_indices)
for p in tq_indices:
    print("Around pos", p, ":", repr(s[max(0, p-20):min(len(s), p+20)]))

