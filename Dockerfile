FROM python:3.12-slim

# 2. 環境変数の設定
# Vertex AIとJAX/XLAのための最適化
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # JAXがGPUメモリをすべて事前に確保するのを防ぐ（他のプロセスと同居する場合などに有効）
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    # OOMを防ぐため、GPUメモリの割り当て割合を指定（必要に応じて調整）
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    # L4 GPU (Compute Capability 8.9) を最適に使うための設定
    TF_CPP_MIN_LOG_LEVEL=2

# システムパッケージと Python 3.12 のインストール
ENV DEBIAN_FRONTEND=noninteractive

# 3. OSの必須パッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python コマンドを python3.12 に紐付け
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python

# Rustのインストール (deckgym-core ビルド用)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"


# Pythonライブラリのインストール
COPY requirements-gcp.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-gcp.txt

# ソースコードをコピー
WORKDIR /app
COPY . /app

# deckgym-core のビルド
RUN cd deckgym-core && cargo build --release --features python \
    && cp target/release/libdeckgym.so python/deckgym/deckgym.so \
    && cp -r python/deckgym /app/ \
    && cp -r python/deckgym_openspiel /app/

# JAX が GPU ライブラリを見つけやすくするための環境変数
# Vertex AI では /usr/local/nvidia/lib64 に GPU ドライバがマウントされる場合がある
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64


# エントリーポイント
ENTRYPOINT ["python", "train.py"]