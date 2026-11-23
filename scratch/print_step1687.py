import json

with open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = json.loads(lines[1687])
print(data.get('content'))

