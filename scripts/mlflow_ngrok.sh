#!/bin/bash

curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

uv pip install --system mlflow

# ポート番号の定義
PORT=5000

# 1. バックグラウンドで MLflow UI を起動
# --host 0.0.0.0 を指定することで外部からの接続を許可します
DRIVE_MLRUNS="/content/drive/MyDrive/R-NaD-PokemonTCGP_experiments/mlruns"

if [ -d "$DRIVE_MLRUNS" ]; then
    echo "Serving MLflow from Google Drive: $DRIVE_MLRUNS"
    nohup mlflow ui --backend-store-uri "$DRIVE_MLRUNS" --host 0.0.0.0 --port $PORT --allowed-hosts "*" > mlflow.log 2>&1 &
else
    echo "Serving MLflow from local ./mlruns"
    nohup mlflow ui --host 0.0.0.0 --port $PORT --allowed-hosts "*" > mlflow.log 2>&1 &
fi

# 2. しばらく待機（起動完了を待つ）
sleep 5

# 3. ngrok でトンネルを作成
# ※ 事前に ngrok config add-authtoken を実行している前提です
echo "MLflow UI を ngrok で公開中..."
ngrok http $PORT
