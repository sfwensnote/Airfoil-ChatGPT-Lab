# -*- coding: utf-8 -*-
"""
本地 LoRA 微调脚本 (使用 Unsloth)
适用于消费级 GPU (RTX 3090/4090 或 Mac M1/M2)
"""

import os
import json
from pathlib import Path

# 检查环境
try:
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️ Unsloth 未安装。请运行以下命令安装:")
    print("   pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")


# ============= 配置 =============
CONFIG = {
    # 模型配置
    "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",  # 或 "unsloth/llama-3-8b-Instruct-bnb-4bit"
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # LoRA 配置
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # 训练配置
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": not torch.cuda.is_bf16_supported() if UNSLOTH_AVAILABLE else True,
    "bf16": torch.cuda.is_bf16_supported() if UNSLOTH_AVAILABLE else False,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42,
    
    # 输出配置
    "output_dir": "./outputs",
}


def load_training_data(data_path: str) -> Dataset:
    """加载训练数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 转换为 Dataset 格式
    formatted_data = []
    for item in data:
        # ShareGPT 格式
        if "conversations" in item:
            text = ""
            if item.get("system"):
                text += f"<|im_start|>system\n{item['system']}<|im_end|>\n"
            for conv in item["conversations"]:
                role = "user" if conv["from"] == "human" else "assistant"
                text += f"<|im_start|>{role}\n{conv['value']}<|im_end|>\n"
            formatted_data.append({"text": text})
        
        # Alpaca 格式
        elif "instruction" in item:
            text = f"<|im_start|>system\n{item.get('instruction', '')}<|im_end|>\n"
            text += f"<|im_start|>user\n{item.get('input', '')}<|im_end|>\n"
            text += f"<|im_start|>assistant\n{item.get('output', '')}<|im_end|>\n"
            formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def train(
    data_path: str,
    output_dir: str = None,
    base_model: str = None,
    num_epochs: int = None,
):
    """执行微调训练"""
    if not UNSLOTH_AVAILABLE:
        print("❌ Unsloth 未安装，无法进行本地训练")
        return
    
    # 合并配置
    config = CONFIG.copy()
    if output_dir:
        config["output_dir"] = output_dir
    if base_model:
        config["base_model"] = base_model
    if num_epochs:
        config["num_train_epochs"] = num_epochs
    
    print(f"🚀 开始微调训练")
    print(f"   基座模型: {config['base_model']}")
    print(f"   训练数据: {data_path}")
    print(f"   输出目录: {config['output_dir']}")
    
    # 1. 加载模型
    print("\n📦 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    
    # 2. 添加 LoRA 适配器
    print("🔧 添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )
    
    # 3. 加载数据
    print("📊 加载训练数据...")
    dataset = load_training_data(data_path)
    print(f"   共 {len(dataset)} 条样本")
    
    # 4. 配置训练器
    print("⚙️ 配置训练器...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            warmup_steps=config["warmup_steps"],
            num_train_epochs=config["num_train_epochs"],
            learning_rate=config["learning_rate"],
            fp16=config["fp16"],
            bf16=config["bf16"],
            logging_steps=config["logging_steps"],
            optim=config["optim"],
            weight_decay=config["weight_decay"],
            lr_scheduler_type=config["lr_scheduler_type"],
            seed=config["seed"],
            output_dir=config["output_dir"],
        ),
    )
    
    # 5. 开始训练
    print("\n🏋️ 开始训练...")
    trainer_stats = trainer.train()
    
    print(f"\n✅ 训练完成！")
    print(f"   训练时间: {trainer_stats.metrics['train_runtime']:.2f} 秒")
    print(f"   每秒样本数: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    
    # 6. 保存模型
    print("\n💾 保存模型...")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    # 可选：保存为 GGUF 格式 (用于 Ollama)
    # model.save_pretrained_gguf(config["output_dir"], tokenizer, quantization_method="q4_k_m")
    
    print(f"\n🎉 模型已保存到: {config['output_dir']}")
    return config["output_dir"]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="本地 LoRA 微调")
    parser.add_argument("--data", type=str, required=True, help="训练数据路径 (JSON)")
    parser.add_argument("--output", type=str, default="./outputs/airfoil-lora", help="输出目录")
    parser.add_argument("--model", type=str, default=None, help="基座模型 (默认: Qwen2.5-7B)")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    
    args = parser.parse_args()
    
    train(
        data_path=args.data,
        output_dir=args.output,
        base_model=args.model,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
