#!/bin/bash
# 快速微调流程脚本

set -e

echo "===== Airfoil Lab 模型微调流程 ====="
echo ""

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: OPENAI_API_KEY 未设置"
    echo "   对于 OpenAI 微调，请设置: export OPENAI_API_KEY='your-key'"
    echo ""
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "📁 工作目录: $(pwd)"
echo ""

# 菜单
echo "请选择操作:"
echo "  1. 转换现有训练数据格式"
echo "  2. 生成更多训练数据 (需要 OpenAI API)"
echo "  3. 使用 OpenAI API 微调"
echo "  4. 本地 LoRA 微调 (需要 GPU)"
echo "  5. 查看微调任务状态"
echo "  6. 退出"
echo ""

read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "📊 转换训练数据..."
        python scripts/convert_data.py
        echo ""
        echo "✅ 数据已转换，保存在 data/ 目录"
        ;;
    
    2)
        echo ""
        read -p "每个角色生成多少条新样本? [10]: " num
        num=${num:-10}
        echo "🔄 生成训练数据 (每角色 $num 条)..."
        python scripts/generate_data.py --num "$num"
        echo ""
        echo "✅ 完成！请再次运行选项 1 转换格式"
        ;;
    
    3)
        echo ""
        echo "🚀 OpenAI API 微调流程"
        echo ""
        
        # 检查数据
        if [ ! -f "data/all_openai.jsonl" ]; then
            echo "❌ 未找到训练数据，请先运行选项 1"
            exit 1
        fi
        
        echo "步骤 1: 上传训练文件..."
        python scripts/openai_finetune.py --action upload --file data/all_openai.jsonl
        
        echo ""
        read -p "请输入上传的文件 ID: " file_id
        
        echo ""
        echo "步骤 2: 创建微调任务..."
        python scripts/openai_finetune.py --action train --file-id "$file_id" --epochs 3
        
        echo ""
        echo "✅ 微调任务已提交！使用选项 5 查看状态"
        ;;
    
    4)
        echo ""
        echo "🖥️  本地 LoRA 微调"
        echo ""
        
        # 检查数据
        if [ ! -f "data/all_sharegpt.json" ]; then
            echo "❌ 未找到训练数据，请先运行选项 1"
            exit 1
        fi
        
        read -p "输出目录 [./outputs/airfoil-lora]: " output_dir
        output_dir=${output_dir:-./outputs/airfoil-lora}
        
        read -p "训练轮数 [3]: " epochs
        epochs=${epochs:-3}
        
        echo ""
        echo "开始训练..."
        python scripts/local_lora_train.py --data data/all_sharegpt.json --output "$output_dir" --epochs "$epochs"
        ;;
    
    5)
        echo ""
        echo "📋 微调任务状态"
        python scripts/openai_finetune.py --action list
        ;;
    
    6)
        echo "👋 再见！"
        exit 0
        ;;
    
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac
