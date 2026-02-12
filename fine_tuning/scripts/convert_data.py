# -*- coding: utf-8 -*-
"""
数据格式转换工具
将训练样本转换为各种微调格式
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

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


def load_samples(role: str, training_data_dir: str = "../training_data") -> List[Dict]:
    """加载指定角色的训练样本"""
    sample_path = Path(training_data_dir) / role / "samples.json"
    if not sample_path.exists():
        print(f"警告: {sample_path} 不存在")
        return []
    
    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_openai_format(role: str, samples: List[Dict]) -> List[Dict]:
    """
    转换为 OpenAI 微调格式
    格式: {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    system_prompt = SYSTEM_PROMPTS.get(role, "你是一个有帮助的AI助手。")
    
    result = []
    for sample in samples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample.get("input", sample.get("instruction", ""))},
            {"role": "assistant", "content": sample.get("output", "")}
        ]
        result.append({"messages": messages})
    
    return result


def convert_to_sharegpt_format(role: str, samples: List[Dict]) -> List[Dict]:
    """
    转换为 ShareGPT 格式 (用于 LLaMA-Factory)
    格式: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}], "system": "..."}
    """
    system_prompt = SYSTEM_PROMPTS.get(role, "你是一个有帮助的AI助手。")
    
    result = []
    for sample in samples:
        conversation = {
            "system": system_prompt,
            "conversations": [
                {"from": "human", "value": sample.get("input", sample.get("instruction", ""))},
                {"from": "gpt", "value": sample.get("output", "")}
            ]
        }
        result.append(conversation)
    
    return result


def convert_to_alpaca_format(role: str, samples: List[Dict]) -> List[Dict]:
    """
    转换为 Alpaca 格式 (用于 Unsloth)
    格式: {"instruction": "...", "input": "...", "output": "..."}
    保留原格式，添加系统提示词到 instruction
    """
    system_prompt = SYSTEM_PROMPTS.get(role, "")
    
    result = []
    for sample in samples:
        alpaca_sample = {
            "instruction": f"{system_prompt}\n\n{sample.get('instruction', '')}",
            "input": sample.get("input", ""),
            "output": sample.get("output", "")
        }
        result.append(alpaca_sample)
    
    return result


def save_jsonl(data: List[Dict], output_path: str):
    """保存为 JSONL 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存 {len(data)} 条数据到 {output_path}")


def save_json(data: List[Dict], output_path: str):
    """保存为 JSON 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(data)} 条数据到 {output_path}")


def main():
    """主函数：转换所有角色的训练数据"""
    script_dir = Path(__file__).parent
    # 训练数据在项目根目录的 training_data 下，而非 fine_tuning/training_data
    project_root = script_dir.parent.parent
    training_data_dir = project_root / "training_data"
    output_dir = project_root / "fine_tuning" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"训练数据目录: {training_data_dir}")
    print(f"输出目录: {output_dir}")

    
    roles = ["concept_mentor", "iteration_engineer", "strategy_analyst"]
    
    # 分别转换每个角色
    all_openai = []
    all_sharegpt = []
    all_alpaca = []
    
    for role in roles:
        print(f"\n处理 {role}...")
        samples = load_samples(role, str(training_data_dir))
        
        if not samples:
            continue
        
        # 转换格式
        openai_data = convert_to_openai_format(role, samples)
        sharegpt_data = convert_to_sharegpt_format(role, samples)
        alpaca_data = convert_to_alpaca_format(role, samples)
        
        # 累积合并
        all_openai.extend(openai_data)
        all_sharegpt.extend(sharegpt_data)
        all_alpaca.extend(alpaca_data)
        
        # 单独保存每个角色
        save_jsonl(openai_data, str(output_dir / f"{role}_openai.jsonl"))
    
    # 保存合并数据
    save_jsonl(all_openai, str(output_dir / "all_openai.jsonl"))
    save_json(all_sharegpt, str(output_dir / "all_sharegpt.json"))
    save_json(all_alpaca, str(output_dir / "all_alpaca.json"))
    
    print(f"\n✅ 转换完成！共 {len(all_openai)} 条训练样本")
    print(f"   - OpenAI 格式: {output_dir / 'all_openai.jsonl'}")
    print(f"   - ShareGPT 格式: {output_dir / 'all_sharegpt.json'}")
    print(f"   - Alpaca 格式: {output_dir / 'all_alpaca.json'}")


if __name__ == "__main__":
    main()
