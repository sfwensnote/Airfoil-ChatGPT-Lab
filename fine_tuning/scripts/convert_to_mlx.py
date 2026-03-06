# -*- coding: utf-8 -*-
"""
将 Alpaca 格式数据转换为 MLX 微调所需的 ChatML JSONL 格式
"""

import json
import os
from pathlib import Path

# 角色到系统提示词的映射
SYSTEM_PROMPTS = {
    "concept_mentor": """你是一位专家级的空气动力学教育者，专门帮助学生理解翼型设计的基础理论。

你的教学风格：
- 采用苏格拉底式问答法：多问启发性问题，引导学生思考
- 使用生动的类比让抽象概念变得直观
- 保持耐心和鼓励的态度

你的知识领域：伯努利原理、边界层理论、翼型参数、NACA命名系统、雷诺数效应和失速机理。""",

    "iteration_engineer": """你是一位经验丰富的翼型设计工程师，专门帮助用户优化翼型参数以达成特定的性能目标。

你的工作方式：
1. 分析当前状态
2. 明确优化目标
3. 提出具体建议
4. 以 JSON 格式输出建议的新参数

使用 <thought>...</thought> 标签展示你的分析过程。""",

    "strategy_analyst": """你是一位资深的数据分析师，专门帮助用户回顾和分析他们的翼型设计迭代历史，总结规律和教训。

你的分析方法：
1. 纵览全局：查看完整设计历史
2. 识别转折点：找出关键的正向/负向迭代
3. 归纳规律：总结有效和无效的调整
4. 提供洞见：指出设计盲点"""
}


def convert_alpaca_to_chatml(input_path: str, output_path: str):
    """将 Alpaca 格式数据转换为 ChatML JSONL 格式"""
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chatml_data = []
    
    for item in data:
        instruction = item.get("instruction", "")
        user_input = item.get("input", "")
        output = item.get("output", "")
        
        # 尝试从 instruction 中识别角色
        system_prompt = ""
        for role, prompt in SYSTEM_PROMPTS.items():
            if prompt in instruction:
                system_prompt = prompt
                # 从 instruction 中去掉系统提示词部分
                instruction = instruction.replace(prompt, "").strip()
                break
        
        # 如果没有匹配到系统提示词，使用 instruction 本身作为系统提示
        if not system_prompt:
            system_prompt = instruction
            user_content = user_input
        else:
            # 合并剩余的 instruction 和 input
            user_content = f"{instruction}\n{user_input}".strip() if instruction else user_input
        
        messages = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
        }
        chatml_data.append(messages)
    
    # 写入 JSONL 格式
    with open(output_path, "w", encoding="utf-8") as f:
        for item in chatml_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 转换完成: {len(chatml_data)} 条样本 → {output_path}")
    return len(chatml_data)


def split_train_valid(input_path: str, train_path: str, valid_path: str, valid_ratio: float = 0.1):
    """将数据拆分为训练集和验证集"""
    import random
    random.seed(42)
    
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    n_valid = max(1, int(len(lines) * valid_ratio))
    
    valid_lines = lines[:n_valid]
    train_lines = lines[n_valid:]
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    
    with open(valid_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)
    
    print(f"📊 数据拆分: {len(train_lines)} 训练 + {len(valid_lines)} 验证")


def main():
    project_root = Path(__file__).parent.parent.parent
    
    # 输入路径 (Alpaca 格式)
    alpaca_path = project_root / "fine_tuning" / "data" / "all_alpaca.json"
    
    # 输出路径
    output_dir = project_root / "fine_tuning" / "data" / "mlx"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chatml_path = output_dir / "all_chatml.jsonl"
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    
    print(f"📂 输入: {alpaca_path}")
    print(f"📂 输出: {output_dir}")
    
    # Step 1: 转换格式
    total = convert_alpaca_to_chatml(str(alpaca_path), str(chatml_path))
    
    # Step 2: 拆分训练/验证
    split_train_valid(str(chatml_path), str(train_path), str(valid_path))
    
    print(f"\n🎉 数据准备完成！共 {total} 条样本")
    print(f"   训练集: {train_path}")
    print(f"   验证集: {valid_path}")


if __name__ == "__main__":
    main()
