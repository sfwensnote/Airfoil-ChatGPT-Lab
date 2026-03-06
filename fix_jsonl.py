import json

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        data = json.loads(line)
        for msg in data['messages']:
            if not isinstance(msg['content'], str):
                msg['content'] = json.dumps(msg['content'], ensure_ascii=False)
        fixed_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

fix_file('/Users/wensifan/bot-remote-windows/fine_tuning/data/mlx/train.jsonl')
fix_file('/Users/wensifan/bot-remote-windows/fine_tuning/data/mlx/valid.jsonl')
print("Fixed JSONL files")
