import json

with open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print('Total lines:', len(lines))
for l in lines[-10:]:
    data = json.loads(l)
    print(f"Step {data.get('step_index')}, source: {data.get('source')}, type: {data.get('type')}")
    if 'content' in data:
        c = data['content']
        if isinstance(c, str):
            print('  Content preview:', repr(c[:200]))

