# ベースイメージ (JAX + TPU対応のものを選ぶ)
FROM python:3.10-slim

# 必要なシステムパッケージとRustをインストール
RUN apt-get update && apt-get install -y \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Rustのインストール
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Pythonライブラリのインストール
COPY requirements-gcp-tpu.txt .
RUN pip install --no-cache-dir -r requirements-gcp-tpu.txt

# ソースコードをコピー
WORKDIR /app
COPY . /app

# deckgym-core のビルド (ここが重要)
RUN cd deckgym-core && cargo build --release --features python \
    && cp target/release/libdeckgym.so python/deckgym/deckgym.so \
    && cp -r python/deckgym /app/ \
    && cp -r python/deckgym_openspiel /app/

# エントリーポイント
ENTRYPOINT ["python", "train.py"]