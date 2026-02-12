#!/bin/bash
# Airfoil Lab 一键启动脚本 (macOS/Linux)
# 同时启动后端 (FastAPI) 和前端 (Next.js)

echo "🚀 Starting Airfoil Lab..."

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查依赖
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python first."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install Node.js first."
    exit 1
fi

# 启动后端
echo "📡 Starting backend (port 8000)..."
cd "$SCRIPT_DIR"
python3 -m uvicorn backend:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# 等待后端启动
sleep 2

# 启动前端
echo "🌐 Starting frontend (port 3000)..."
cd "$SCRIPT_DIR/airfoil-lab-react"
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

# 等待前端启动
sleep 3

echo ""
echo "✅ Airfoil Lab is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both services..."

# 捕获退出信号
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# 保持脚本运行
wait
