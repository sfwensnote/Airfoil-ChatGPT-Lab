# -*- coding: utf-8 -*-
"""
OpenAI API 微调脚本
使用 OpenAI Fine-tuning API 微调 GPT-4o-mini
"""

import os
import time
import json
from pathlib import Path
from openai import OpenAI

# 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_MODEL = "gpt-4o-mini-2024-07-18"  # 支持微调的模型

# 可选：自定义 API Base (如使用代理)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)


def get_client():
    """创建 OpenAI 客户端"""
    if not OPENAI_API_KEY:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_API_BASE:
        kwargs["base_url"] = OPENAI_API_BASE
    
    return OpenAI(**kwargs)


def upload_training_file(client: OpenAI, file_path: str) -> str:
    """上传训练文件到 OpenAI"""
    print(f"📤 上传训练文件: {file_path}")
    
    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    file_id = response.id
    print(f"✅ 文件上传成功，ID: {file_id}")
    return file_id


def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    model: str = BASE_MODEL,
    suffix: str = "airfoil-lab",
    n_epochs: int = 3,
    validation_file_id: str = None
) -> str:
    """创建微调任务"""
    print(f"🚀 创建微调任务...")
    print(f"   基座模型: {model}")
    print(f"   训练轮数: {n_epochs}")
    print(f"   后缀: {suffix}")
    
    kwargs = {
        "training_file": training_file_id,
        "model": model,
        "suffix": suffix,
        "hyperparameters": {
            "n_epochs": n_epochs,
        }
    }
    
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    
    job = client.fine_tuning.jobs.create(**kwargs)
    
    print(f"✅ 微调任务已创建，ID: {job.id}")
    return job.id


def wait_for_job(client: OpenAI, job_id: str, poll_interval: int = 60):
    """等待微调任务完成"""
    print(f"\n⏳ 等待微调任务完成 (每 {poll_interval} 秒检查一次)...")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        print(f"   状态: {status}")
        
        if status == "succeeded":
            print(f"\n✅ 微调成功！")
            print(f"   新模型 ID: {job.fine_tuned_model}")
            return job.fine_tuned_model
        
        elif status == "failed":
            print(f"\n❌ 微调失败")
            print(f"   错误: {job.error}")
            return None
        
        elif status == "cancelled":
            print(f"\n⚠️ 微调被取消")
            return None
        
        time.sleep(poll_interval)


def list_fine_tuning_jobs(client: OpenAI, limit: int = 10):
    """列出最近的微调任务"""
    print(f"\n📋 最近的微调任务:")
    
    jobs = client.fine_tuning.jobs.list(limit=limit)
    
    for job in jobs.data:
        print(f"   - {job.id}: {job.status} ({job.model} -> {job.fine_tuned_model or 'N/A'})")


def test_fine_tuned_model(client: OpenAI, model_id: str, test_prompt: str):
    """测试微调后的模型"""
    print(f"\n🧪 测试微调模型: {model_id}")
    print(f"   输入: {test_prompt[:100]}...")
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": test_prompt}
        ],
        max_tokens=500
    )
    
    output = response.choices[0].message.content
    print(f"   输出: {output[:200]}...")
    return output


def main():
    """主流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI 模型微调")
    parser.add_argument("--action", choices=["upload", "train", "status", "list", "test"], required=True)
    parser.add_argument("--file", type=str, help="训练数据文件路径 (JSONL)")
    parser.add_argument("--file-id", type=str, help="已上传的文件 ID")
    parser.add_argument("--job-id", type=str, help="微调任务 ID")
    parser.add_argument("--model-id", type=str, help="微调后的模型 ID")
    parser.add_argument("--role", type=str, choices=["concept_mentor", "iteration_engineer", "strategy_analyst", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--prompt", type=str, default="什么是雷诺数？", help="测试提示词")
    
    args = parser.parse_args()
    client = get_client()
    
    if args.action == "upload":
        # 上传文件
        if not args.file:
            # 默认使用转换后的数据
            data_dir = Path(__file__).parent.parent / "data"
            if args.role == "all":
                args.file = str(data_dir / "all_openai.jsonl")
            else:
                args.file = str(data_dir / f"{args.role}_openai.jsonl")
        
        file_id = upload_training_file(client, args.file)
        print(f"\n📝 下一步使用: --action train --file-id {file_id}")
    
    elif args.action == "train":
        if not args.file_id:
            raise ValueError("请提供 --file-id")
        
        suffix = f"airfoil-{args.role}" if args.role != "all" else "airfoil-all"
        job_id = create_fine_tuning_job(
            client,
            args.file_id,
            suffix=suffix,
            n_epochs=args.epochs
        )
        
        print(f"\n📝 查看状态: --action status --job-id {job_id}")
        
        # 可选：等待完成
        # model_id = wait_for_job(client, job_id)
    
    elif args.action == "status":
        if args.job_id:
            job = client.fine_tuning.jobs.retrieve(args.job_id)
            print(f"任务 ID: {job.id}")
            print(f"状态: {job.status}")
            print(f"模型: {job.fine_tuned_model or 'N/A'}")
            print(f"已训练 tokens: {job.trained_tokens or 0}")
        else:
            list_fine_tuning_jobs(client)
    
    elif args.action == "list":
        list_fine_tuning_jobs(client)
    
    elif args.action == "test":
        if not args.model_id:
            raise ValueError("请提供 --model-id")
        test_fine_tuned_model(client, args.model_id, args.prompt)


if __name__ == "__main__":
    main()
