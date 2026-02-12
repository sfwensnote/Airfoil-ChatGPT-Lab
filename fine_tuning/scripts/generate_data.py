# -*- coding: utf-8 -*-
"""
训练数据扩展生成器
使用 LLM API 生成更多高质量训练样本
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 自定义 API 配置
CUSTOM_API_BASE = os.getenv("LLM_API_URL", "http://49.51.37.239:3006/v1")
CUSTOM_API_KEY = os.getenv("LLM_API_KEY", "sk-VrwOEEFLgjJwSOjH5pHRTDorgf0SmJVQrjK2D1uyjxZcfsrn")
DEFAULT_MODEL = "gpt-4o"  # 或 gpt-4o-mini



# ============= 数据生成模板 =============

CONCEPT_MENTOR_TOPICS = [
    "解释伯努利原理在翼型升力产生中的作用",
    "什么是边界层？层流和湍流边界层有什么区别？",
    "解释翼型的厚度如何影响其空气动力学性能",
    "什么是诱导阻力？它与升力有什么关系？",
    "解释马赫数对翼型性能的影响",
    "什么是临界马赫数？为什么它很重要？",
    "解释翼型的前缘半径如何影响失速特性",
    "什么是层流分离气泡？它如何影响翼型性能？",
    "解释NACA 4位数翼型命名系统",
    "什么是翼型的零升攻角？它由什么决定？",
    "解释翼展对升阻比的影响",
    "什么是展弦比？它为什么重要？",
    "解释翼型后缘形状对气动特性的影响",
    "什么是超临界翼型？它的设计目的是什么？",
    "解释雷诺数效应对小型无人机翼型选择的影响",
]

ITERATION_ENGINEER_SCENARIOS = [
    {"goal": "最大化升阻比", "naca": "2412", "re": 500000, "constraint": "保持厚度不变"},
    {"goal": "提高最大升力系数", "naca": "0012", "re": 100000, "constraint": "无"},
    {"goal": "降低巡航阻力", "naca": "4415", "re": 1000000, "constraint": "升力系数不低于0.5"},
    {"goal": "改善失速特性", "naca": "6412", "re": 200000, "constraint": "无"},
    {"goal": "增加低速性能", "naca": "2410", "re": 80000, "constraint": "保持结构强度"},
    {"goal": "优化高速巡航", "naca": "0010", "re": 2000000, "constraint": "马赫数<0.7"},
    {"goal": "平衡升力和阻力", "naca": "4412", "re": 300000, "constraint": "无"},
    {"goal": "最大化爬升率", "naca": "2415", "re": 400000, "constraint": "无"},
]

STRATEGY_REVIEW_PATTERNS = [
    {
        "pattern": "厚度陷阱",
        "description": "反复增加厚度导致阻力增加",
        "history": [
            {"naca": "2412", "ld": 45},
            {"naca": "2415", "ld": 38},
            {"naca": "2418", "ld": 32},
        ]
    },
    {
        "pattern": "弯度探索成功",
        "description": "逐步增加弯度找到最优点",
        "history": [
            {"naca": "0012", "ld": 35},
            {"naca": "2412", "ld": 48},
            {"naca": "4412", "ld": 52},
            {"naca": "6412", "ld": 49},
        ]
    },
    {
        "pattern": "过度优化",
        "description": "在小范围内反复调整没有实质进展",
        "history": [
            {"naca": "4412", "ld": 50},
            {"naca": "4413", "ld": 49.5},
            {"naca": "4411", "ld": 50.2},
            {"naca": "4412", "ld": 50.1},
        ]
    },
]


def generate_concept_samples(client, num_samples: int = 10) -> List[Dict]:
    """使用 GPT-4 生成概念学习样本"""
    samples = []
    
    for topic in random.sample(CONCEPT_MENTOR_TOPICS, min(num_samples, len(CONCEPT_MENTOR_TOPICS))):
        print(f"  生成: {topic[:30]}...")
        
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """你是一位专家级的空气动力学教育者。请生成一个高质量的教学对话样本，以 json 格式返回。
{
  "instruction": "解释...",
  "input": "学生的问题",
  "output": "你的回答（采用苏格拉底式教学，使用类比，以问题结尾引导深入学习）"
}

回答风格：
- 循循善诱，不直接给答案
- 使用生动的类比
- 以后续问题结尾
- 300-500字"""
                },
                {
                    "role": "user",
                    "content": f"请为以下主题生成教学对话样本：{topic}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        
        try:
            sample = json.loads(response.choices[0].message.content)
            samples.append(sample)
        except json.JSONDecodeError:
            continue
    
    return samples


def generate_iteration_samples(client, num_samples: int = 10) -> List[Dict]:
    """使用 GPT-4 生成迭代工程样本"""
    samples = []
    
    for scenario in random.sample(ITERATION_ENGINEER_SCENARIOS, min(num_samples, len(ITERATION_ENGINEER_SCENARIOS))):
        print(f"  生成: {scenario['goal']}...")
        
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """你是一位翼型设计工程师。请生成一个高质量的设计优化对话样本，以 json 格式返回。
{
  "instruction": "根据用户的优化目标，给出具体的翼型参数调整建议。",
  "input": "用户状态: NACA XXXX, Re=XXXXXX, 当前性能。用户目标: XXX",
  "output": "使用 <thought>...</thought> 展示分析过程，然后给出建议，最后输出 JSON 参数"
}

输出格式：
- 先用 <thought> 分析
- 然后给出1-3个具体调整建议
- 最后输出建议的新参数 JSON"""
                },
                {
                    "role": "user",
                    "content": f"场景：当前翼型 NACA {scenario['naca']}，Re={scenario['re']}，目标：{scenario['goal']}，约束：{scenario['constraint']}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        
        try:
            sample = json.loads(response.choices[0].message.content)
            samples.append(sample)
        except json.JSONDecodeError:
            continue
    
    return samples


def generate_strategy_samples(client, num_samples: int = 5) -> List[Dict]:
    """使用 GPT-4 生成策略分析样本"""
    samples = []
    
    for pattern in STRATEGY_REVIEW_PATTERNS:
        print(f"  生成: {pattern['pattern']}...")
        
        history_str = "\n".join([f"{i+1}. NACA {h['naca']}, L/D={h['ld']}" for i, h in enumerate(pattern['history'])])
        
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """你是一位数据分析师。请生成一个高质量的设计历史分析样本，以 json 格式返回。
{
  "instruction": "分析用户的设计迭代历史，总结成功或失败的原因。",
  "input": "用户设计历史 (按时间顺序): ...",
  "output": "分析报告，包含表格、趋势分析、核心发现、建议"
}

输出格式：
- 使用 Markdown 表格对比迭代
- 识别关键转折点
- 总结规律和教训
- 给出下一步建议"""
                },
                {
                    "role": "user",
                    "content": f"分析模式：{pattern['pattern']}\n描述：{pattern['description']}\n历史记录：\n{history_str}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        
        try:
            sample = json.loads(response.choices[0].message.content)
            samples.append(sample)
        except json.JSONDecodeError:
            continue
    
    return samples


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成扩展训练数据")
    parser.add_argument("--role", choices=["concept_mentor", "iteration_engineer", "strategy_analyst", "all"], default="all")
    parser.add_argument("--num", type=int, default=10, help="每个角色生成的样本数")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    if not OPENAI_AVAILABLE:
        print("❌ 需要安装 openai: pip install openai")
        return
    
    # 使用自定义 API 配置
    print(f"📡 使用 API: {CUSTOM_API_BASE}")
    print(f"📦 使用模型: {DEFAULT_MODEL}")
    
    client = OpenAI(
        api_key=CUSTOM_API_KEY,
        base_url=CUSTOM_API_BASE,
    )
    
    # 输出目录 - 项目根目录的 training_data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = Path(args.output) if args.output else project_root / "training_data"
    
    roles_to_generate = ["concept_mentor", "iteration_engineer", "strategy_analyst"] if args.role == "all" else [args.role]
    
    for role in roles_to_generate:
        print(f"\n🔄 生成 {role} 数据...")
        
        if role == "concept_mentor":
            samples = generate_concept_samples(client, args.num)
        elif role == "iteration_engineer":
            samples = generate_iteration_samples(client, args.num)
        else:
            samples = generate_strategy_samples(client, args.num)
        
        # 保存
        role_dir = output_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        
        # 追加到现有数据
        existing_path = role_dir / "samples.json"
        existing_samples = []
        if existing_path.exists():
            with open(existing_path, "r", encoding="utf-8") as f:
                existing_samples = json.load(f)
        
        all_samples = existing_samples + samples
        
        with open(existing_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 已生成 {len(samples)} 条新样本，总计 {len(all_samples)} 条")
    
    print("\n🎉 数据生成完成！请运行 convert_data.py 转换格式")


if __name__ == "__main__":
    main()

