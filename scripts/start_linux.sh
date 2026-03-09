#!/bin/bash

# ==============================================================================
#                 遥感图像智能解译系统 - 一键启动脚本 (Linux/macOS)
# ==============================================================================

# --- 1. 获取脚本所在的绝对路径，确保所有相对路径都正确 ---
BASE_DIR=$(cd "$(dirname "$0")" && pwd)
echo "项目根目录: $BASE_DIR"
echo ""

# --- 2. 定义所有需要的路径变量 ---
# 后端脚本路径
BACKEND_CONVERT_SCRIPT="$BASE_DIR/backend/convert_image.py"
BACKEND_PREDICT_SCRIPT="$BASE_DIR/backend/predict.py" # 注意，你的文件名是 predict.py

# 前端项目路径
FRONTEND_DIR="$BASE_DIR/frontend"

# Conda 环境路径
CONDA_ENV_DIR="$BASE_DIR/deploy_env"

# --- 3. 检查关键文件和目录是否存在 ---
if [ ! -d "$CONDA_ENV_DIR" ]; then
    echo "❌ 错误: Conda 环境目录 '$CONDA_ENV_DIR' 未找到。"
    echo "请确保你已经将 'deploy_env.tar.gz' 解压到了项目根目录。"
    exit 1
fi
if [ ! -f "$BACKEND_PREDICT_SCRIPT" ]; then
    echo "❌ 错误: 预测脚本 '$BACKEND_PREDICT_SCRIPT' 未找到。"
    exit 1
fi
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ 错误: 前端目录 '$FRONTEND_DIR' 未找到。"
    exit 1
fi

echo "✅ 所有关键路径检查通过。"
echo ""

# --- 4. 启动所有服务 ---
echo "=========================================================="
echo "           🚀 正在启动所有服务，请稍候..."
echo "=========================================================="
echo ""

# 启动图片转换服务
echo "[1/3] 正在后台启动图片转换服务 (端口 5001)..."
conda run -p "$CONDA_ENV_DIR" python "$BACKEND_CONVERT_SCRIPT" &
CONVERT_PID=$!
echo "  - 转换服务已启动，进程 ID: $CONVERT_PID"
echo ""

# 启动模型预测服务
echo "[2/3] 正在后台启动模型预测服务 (端口 5000)..."
# 注意：我们将模型路径通过命令行参数传递给脚本
conda run -p "$CONDA_ENV_DIR" python "$BACKEND_PREDICT_SCRIPT" --model_dir "$BASE_DIR/model_and_hra" &> predict.log &
PREDICT_PID=$!
echo "  - 预测服务已启动，进程 ID: $PREDICT_PID"
echo ""

# 等待后端模型加载
echo "⏳ 正在等待后端模型预热 (约 15-30 秒)..."
sleep 20 # 等待20秒，如果你的模型加载很慢，可以适当增加这个时间

# 启动前端开发服务器
echo "[3/3] 正在启动前端 React 应用..."
cd "$FRONTEND_DIR"
npm start &
FRONTEND_PID=$!
echo "  - 前端服务已启动，进程 ID: $FRONTEND_PID"
echo ""

# --- 5. 设置清理函数，以便在关闭脚本时能杀死所有后台进程 ---
cleanup() {
    echo ""
    echo "=========================================================="
    echo "🛑 收到关闭信号，正在停止所有服务..."
    echo "=========================================================="
    # 使用 kill 命令并抑制错误输出（以防某个进程已经不存在）
    kill $CONVERT_PID 2>/dev/null
    kill $PREDICT_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ 所有服务已停止。"
}

trap cleanup EXIT

# --- 6. 完成提示并等待 ---
echo "=========================================================="
echo "🎉 启动完成！浏览器应该会自动打开应用。"
echo "所有服务正在后台运行。"
echo "如需停止所有服务，请按 Ctrl+C 或直接关闭此终端窗口。"
echo "=========================================================="

# 等待所有后台任务完成（实际上是无限等待，直到脚本被中断）
wait