#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)

# Determine library paths for LD_LIBRARY_PATH (same as train.sh)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cublas; print(nvidia.cublas.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cufft; print(nvidia.cufft.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cusolver; print(nvidia.cusolver.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_cupti; print(nvidia.cuda_cupti.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cusparse; print(nvidia.cusparse.__path__[0])")/lib


# 1. deckgym-core ディレクトリに移動
cd deckgym-core
# 2. ビルドして wheel ファイルを作成
#    --release: 最適化を有効にする
#    --features python: Pythonバインディングを有効にする
#    --interpreter: プロジェクトの仮想環境のPythonを指定
python3 -m maturin build --release --features python --interpreter ../.venv/bin/python3
# 3. 作成された wheel を「強制再インストール」で適用
python3 -m pip install target/wheels/deckgym-*.whl --force-reinstall

cd ..

# JAXが動かなくなった場合のみ実行
python3 -m pip install numpy==1.26.4 scipy==1.12.0

python src/battle.py \
    --checkpoint "checkpoints/" \
    --deck_id_1 "train_data/8acd216f.txt" \
    --deck_id_2 "train_data/cacc7f16.txt" \
    --device "cpu" \
    "$@"

#     --checkpoint "saved_model/74e1ccf674b2b3504a0112d84b4601bb741968be/199" \