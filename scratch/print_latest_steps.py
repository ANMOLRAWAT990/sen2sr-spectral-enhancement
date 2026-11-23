import json

with open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx in range(1670, len(lines)):
    data = json.loads(lines[idx])
    t = data.get('type')
    s = data.get('source')
    step = data.get('step_index')
    content = repr(data.get('content'))[:150]
    print(f"[{idx}] Step {step}, src={s}, type={t}: {content}")

