# run.sh

#!/bin/bash

# 解析參數：預設空，如果有 --resume 傳入就使用
RESUME_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --resume)
      RESUME_PATH="$2"
      shift 2
      ;;
    *)
      echo "❌ 不認識的參數: $1"
      exit 1
      ;;
  esac
done

# 取得當天日期
DATE=$(date +"%Y%m%d")

# 自動計算當天實驗次數
COUNT=$(ls ./EXP 2>/dev/null | grep "run_${DATE}" | wc -l)
COUNT=$(printf "%02d" $((COUNT + 1)))

# 設定 EXP 資料夾
EXP_DIR=./EXP/run_${DATE}_${COUNT}
mkdir -p ${EXP_DIR}/TB
mkdir -p ${EXP_DIR}/checkpoint
mkdir -p ${EXP_DIR}/logs

# 匯出環境變數給 train.py
export EXP_DIR=${EXP_DIR}
export TB_DIR=${EXP_DIR}/TB
export CKPT_DIR=${EXP_DIR}/checkpoint
export RESUME_PATH=${RESUME_PATH}

# TensorBoard 路徑變數
TB_PORT=6006
EXTERNAL_PORT=8812

# 自動偵測最新 EXP 目錄供 TensorBoard 使用
LATEST_EXP_DIR=$(ls -td ./EXP/run_* | head -n 1)

# 啟動 TensorBoard
echo "🚀 啟動 TensorBoard..."
tensorboard --logdir=${LATEST_EXP_DIR} --port=${TB_PORT} --host=0.0.0.0 &

sleep 3

echo "📊 TensorBoard 已啟動！請在瀏覽器打開："
echo "👉 http://$(hostname -I | awk '{print $1}'):${EXTERNAL_PORT}"
echo "📂 當前 EXP 目錄: ${EXP_DIR}"
echo "📂 TensorBoard 目錄: ${LATEST_EXP_DIR}"
echo "📂 Checkpoint 目錄: ${CKPT_DIR}"

# 開始模型訓練並保存終端 log
LOG_FILE=${EXP_DIR}/logs/run_${DATE}_${COUNT}.log

if [ -n "$RESUME_PATH" ]; then
  echo "🔄 恢復訓練模式，載入 checkpoint: $RESUME_PATH"
  python -m torch.distributed.launch --nproc_per_node=1 train.py --devices 0 --resume $RESUME_PATH | tee ${LOG_FILE}
else
  python -m torch.distributed.launch --nproc_per_node=1 train.py --devices 0 | tee ${LOG_FILE}
fi

# 訓練完成提示最新 checkpoint
LATEST_CKPT=$(ls -t ${EXP_DIR}/checkpoint/*.pth | head -n 1)
echo ""
echo "✅ 訓練完成！最新的 checkpoint 儲存在："
echo "${LATEST_CKPT}"
echo "🚀 你可以用這個路徑 resume 訓練哦！"