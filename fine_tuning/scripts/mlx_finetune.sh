#!/bin/bash
# MLX LoRA 微调脚本
# 使用 mlx-lm 框架在 Apple Silicon 上微调 Qwen2.5-3B-Instruct
#
# 用法: bash fine_tuning/scripts/mlx_finetune.sh
#
# 参考: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/fine_tuning/data/mlx"
OUTPUT_DIR="$PROJECT_ROOT/fine_tuning/outputs/mlx-lora"

# 基座模型 (4-bit 量化版, 适合 16GB Mac)
BASE_MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit"

echo "================================================"
echo "  翼型教学 AI - MLX LoRA 微调"
echo "================================================"
echo "基座模型: $BASE_MODEL"
echo "训练数据: $DATA_DIR/train.jsonl"
echo "验证数据: $DATA_DIR/valid.jsonl"
echo "输出目录: $OUTPUT_DIR"
echo "================================================"

# Step 0: 检查数据文件
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "❌ 训练数据不存在，先运行数据转换:"
    echo "   python3 fine_tuning/scripts/convert_to_mlx.py"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
VALID_COUNT=$(wc -l < "$DATA_DIR/valid.jsonl")
echo "训练样本: $TRAIN_COUNT 条"
echo "验证样本: $VALID_COUNT 条"
echo ""

# Step 1: 执行 LoRA 微调
echo "🏋️ 开始 LoRA 微调训练..."
mlx_lm.lora \
    --model "$BASE_MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$OUTPUT_DIR" \
    --iters 200 \
    --batch-size 1 \
    --num-layers 16 \
    --learning-rate 1e-5 \
    --val-batches 2 \
    --steps-per-eval 50 \
    --steps-per-report 10 \
    --save-every 50

echo ""
echo "✅ 微调完成！Adapter 保存在: $OUTPUT_DIR"

# Step 2: 融合 adapter 到基座模型
echo ""
echo "🔗 融合 adapter 到基座模型..."
FUSED_DIR="$PROJECT_ROOT/fine_tuning/outputs/mlx-fused"
mlx_lm.fuse \
    --model "$BASE_MODEL" \
    --adapter-path "$OUTPUT_DIR" \
    --save-path "$FUSED_DIR"

echo "✅ 融合完成！模型保存在: $FUSED_DIR"

# Step 3: 快速测试
echo ""
echo "🧪 快速测试融合后模型..."
mlx_lm.generate \
    --model "$FUSED_DIR" \
    --prompt "什么是雷诺数？它对翼型设计有什么影响？" \
    --max-tokens 200

echo ""
echo "================================================"
echo "🎉 全部完成！"
echo ""
echo "下一步选项:"
echo "  1. 导出 GGUF 并部署到 Ollama (推荐):"
echo "     bash fine_tuning/scripts/deploy_ollama.sh"
echo "  2. 直接使用 MLX 推理:"
echo "     mlx_lm.generate --model $FUSED_DIR --prompt '你的问题'"
echo "================================================"
