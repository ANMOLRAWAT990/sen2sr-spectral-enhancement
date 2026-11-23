import json

with open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx, l in enumerate(lines):
    data = json.loads(l)
    if data.get('type') == 'USER_INPUT':
        print(f"Index {idx}, Step {data.get('step_index')}: {repr(data.get('content'))[:120]}")

