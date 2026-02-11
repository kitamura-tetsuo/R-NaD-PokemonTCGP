import haiku as hk
import jax
import jax.numpy as jnp

class ResidualBlock(hk.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def __call__(self, x):
        input_val = x
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)
        return input_val + x

class DeckGymNet(hk.Module):
    def __init__(self, num_actions, hidden_size=256, num_blocks=4):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

    def __call__(self, x):
        # Flatten input if needed, though usually expected (B, obs_dim)
        if x.ndim > 2:
            x = jnp.reshape(x, (x.shape[0], -1))

        # Torso
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResidualBlock(self.hidden_size)(x)

        # Heads
        policy_logits = hk.Linear(self.num_actions)(x)
        value = hk.Linear(1)(x)
        return policy_logits, value


class TransformerBlock(hk.Module):
    def __init__(self, num_heads, key_size, hidden_size, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training=False):
        # x: (B, SeqLen, EmbedDim)
        
        # Self-Attention
        attn_out = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=hk.initializers.VarianceScaling(2.0),
        )(x, x, x)
        
        # Add & Norm
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + attn_out)
        
        # MLP
        mlp_out = hk.nets.MLP(
            [self.hidden_size * 2, self.hidden_size],
            activation=jax.nn.gelu
        )(x)
        
        # Add & Norm
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + mlp_out)
        
        return x

class TransformerNet(hk.Module):
    def __init__(self, num_actions, hidden_size=64, num_blocks=2, num_heads=4, seq_len=16):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size # Embed Dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.seq_len = seq_len

    def __call__(self, x, is_training=False):
        # x: (B, ObsDim)
        if x.ndim > 2:
            x = jnp.reshape(x, (x.shape[0], -1))
            
        batch_size = x.shape[0]
        obs_dim = x.shape[1]
        
        # Feature Projection: (B, ObsDim) -> (B, SeqLen * EmbedDim)
        # We project the flat observation into a latent sequence
        projection_size = self.seq_len * self.hidden_size
        x = hk.Linear(projection_size)(x)
        
        # Reshape to sequence: (B, SeqLen, EmbedDim)
        x = jnp.reshape(x, (batch_size, self.seq_len, self.hidden_size))
        
        # Add learned position embeddings (optional but good for latent sequence)
        # For simplicity in this "Latent Transformer" on flat data, we might skip or add them.
        # Let's add them.
        pos_emb = hk.get_parameter("pos_emb", [self.seq_len, self.hidden_size], init=hk.initializers.TruncatedNormal())
        x = x + pos_emb
        
        # Transformer Blocks
        for _ in range(self.num_blocks):
            x = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.hidden_size // self.num_heads,
                hidden_size=self.hidden_size
            )(x, is_training)
            
        # Global Pooling (Mean) -> (B, EmbedDim)
        x = jnp.mean(x, axis=1)
        
        # Heads
        # Policy
        policy_logits = hk.Linear(self.num_actions)(x)
        
        # Value
        value = hk.Linear(1)(x)
        
        return policy_logits, value

class PrecomputedEmbedding(hk.Module):
    """事前計算済み特徴量テーブルからベクトルをLookupするモジュール"""
    def __init__(self, embedding_matrix, output_dim, name=None):
        super().__init__(name=name)
        self.embedding_matrix = embedding_matrix # shape: (TotalCards, FeatureDim)
        self.output_dim = output_dim

    def __call__(self, card_ids):
        # card_ids: (Batch, NumSlots) の整数配列
        
        # 1. テーブルを定数パラメータとして定義 (学習しない)
        table = hk.get_parameter(
            "feature_table",
            shape=self.embedding_matrix.shape,
            init=hk.initializers.Constant(self.embedding_matrix)
        )
        table = jax.lax.stop_gradient(table) # 固定する
        
        # 2. IDに対応するベクトルを引く (Lookup)
        # 負のID (-1.0) は null として扱う (index 0 が null と想定するか、あるいは clip する)
        # ここでは clip(0, num_cards-1) してから、有効なIDかどうかでマスクするのが安全
        num_cards = self.embedding_matrix.shape[0]
        valid_mask = (card_ids >= 0) & (card_ids < num_cards)
        safe_ids = jnp.where(valid_mask, card_ids, 0)
        
        # x shape: (Batch, NumSlots, FeatureDim)
        x = jnp.take(table, safe_ids, axis=0)
        
        # 無効なIDの部分をゼロ埋め
        x = jnp.where(valid_mask[..., None], x, 0.0)
        
        # 3. Transformerの次元(hidden_size)に合わせて圧縮/変換
        x = hk.Linear(self.output_dim)(x)
        return x

class HybridEmbedding(hk.Module):
    """汎用ベクトル(Static) + 補正パッチ(Residual) を組み合わせた埋め込み"""
    def __init__(self, pretrained_matrix, output_dim, name=None):
        super().__init__(name=name)
        self.pretrained_matrix = pretrained_matrix # (NumCards, FeatureDim)
        self.output_dim = output_dim

    def __call__(self, card_ids):
        # 1. Static Path (テキストの意味)
        #    PrecomputedEmbedding 内で Linear 投影まで行われる
        static_proj = PrecomputedEmbedding(
            self.pretrained_matrix, 
            self.output_dim, 
            name="static_path"
        )(card_ids)

        # 2. Residual Path (IDごとの例外補正)
        #    w_init=0 で初期化し、学習が進むにつれて例外を吸収する
        num_cards = self.pretrained_matrix.shape[0]
        valid_mask = (card_ids >= 0) & (card_ids < num_cards)
        safe_ids = jnp.where(valid_mask, card_ids, 0)

        residual_emb = hk.Embed(
            vocab_size=num_cards,
            embed_dim=self.output_dim,
            w_init=hk.initializers.Constant(0.0), 
            name="residual_path"
        )(safe_ids)

        # 無効なIDの部分をゼロ埋め
        residual_emb = jnp.where(valid_mask[..., None], residual_emb, 0.0)

        # 3. Add (統合)
        return static_proj + residual_emb

class CardTransformerNet(hk.Module):
    def __init__(self, num_actions, embedding_matrix, hidden_size=64, num_blocks=2, num_heads=4):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_matrix = embedding_matrix
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

    def __call__(self, x, is_training=False):
        # x: (Batch, ObsDim)
        batch_size = x.shape[0]
        
        # 1. 情報の抽出
        # 前半 39 dims: ターン情報など
        # その後 8 slots * 24 dims = 192 dims: 盤面カード情報 (カードIDは index 11)
        # その後 10 slots: 手札 (カードID)
        # その後 2 dims: デッキ残り枚数
        # その後 10 slots: 自分トラッシュ (カードID)
        # その後 10 slots: 相手トラッシュ (カードID)
        
        turn_info = x[:, :39]
        board_part = x[:, 39:39+192].reshape(batch_size, 8, 24)
        hand_ids = x[:, 231:241].astype(jnp.int32)
        deck_ids = x[:, 241:261].astype(jnp.int32)
        opp_deck_count = x[:, 261:262]
        discard_ids = x[:, 262:272].astype(jnp.int32)
        opp_discard_ids = x[:, 272:282].astype(jnp.int32)
        
        # 2. 埋め込み (Board)
        board_card_ids = board_part[:, :, 11].astype(jnp.int32)
        board_card_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_board")(board_card_ids)
        
        # 盤面の動的特徴量 (HP, Energy, Statusなど) の抽出
        indices = [i for i in range(24) if i != 11]
        board_features = board_part[:, :, indices]
        board_features = hk.Linear(self.hidden_size, name="lin_board_feat")(board_features)
        board_slot_repr = board_card_emb + board_features
        
        # 3. 埋め込み (Hand, Deck, Discard)
        # これらは動的特徴量がないので、単に埋め込みだけ
        hand_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_hand")(hand_ids)
        deck_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_deck")(deck_ids)
        discard_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_discard")(discard_ids)
        opp_discard_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_opp_discard")(opp_discard_ids)
        
        # 4. Slot Positioning (場所と意味の結合)
        # 各エリアごとの位置埋め込み
        def get_pos_emb(name, num_slots):
            return hk.get_parameter(name, [1, num_slots, self.hidden_size], init=hk.initializers.TruncatedNormal())
            
        board_pos = get_pos_emb("pos_board", 8)
        hand_pos = get_pos_emb("pos_hand", 10)
        deck_pos = get_pos_emb("pos_deck", 20)
        discard_pos = get_pos_emb("pos_discard", 10)
        opp_discard_pos = get_pos_emb("pos_opp_discard", 10)
        
        # トークン列の構築
        tokens = [
            board_slot_repr + board_pos,
            hand_emb + hand_pos,
            deck_emb + deck_pos,
            discard_emb + discard_pos,
            opp_discard_emb + opp_discard_pos
        ]
        x_seq = jnp.concatenate(tokens, axis=1) # (Batch, 58, hidden_size)
        
        # 5. Transformer Blocks
        for i in range(self.num_blocks):
            x_seq = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.hidden_size // self.num_heads,
                hidden_size=self.hidden_size,
                name=f"block_{i}"
            )(x_seq, is_training)
            
        # 6. Global Features 統合
        # 全トークンの平均
        global_summary = jnp.mean(x_seq, axis=1)
        
        # ターン情報とデッキ情報 (自分はID列から計算, 相手は数値) を圧縮
        self_deck_count = jnp.sum(deck_ids >= 0, axis=1, keepdims=True).astype(jnp.float32)
        context_repr = jnp.concatenate([turn_info, self_deck_count, opp_deck_count], axis=-1)
        context_repr = hk.Linear(self.hidden_size, name="lin_context")(context_repr)
        context_repr = jax.nn.relu(context_repr)
        
        final_repr = jnp.concatenate([global_summary, context_repr], axis=-1)
        final_repr = hk.Linear(self.hidden_size, name="lin_final")(final_repr)
        final_repr = jax.nn.relu(final_repr)
        
        # 7. Heads
        policy_logits = hk.Linear(self.num_actions, name="lin_policy")(final_repr)
        value = hk.Linear(1, name="lin_value")(final_repr)
        
        return policy_logits, value
