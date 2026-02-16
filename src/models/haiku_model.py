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

    def __call__(self, x, mask=None):
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
        
        if mask is not None:
             inf_mask = (1.0 - mask) * -1e9
             policy_logits = policy_logits + inf_mask

        value = hk.Linear(2)(x)
        return policy_logits, value


class TransformerBlock(hk.Module):
    def __init__(self, num_heads, key_size, hidden_size, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training=False):
        # x: (B, SeqLen, D)
        d = x.shape[-1]
        
        # Self-Attention
        attn_out = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=hk.initializers.VarianceScaling(2.0),
            model_size=d,
        )(x, x, x)
        
        # Add & Norm
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + attn_out)
        
        # MLP: Output dimension is adjusted to d to enable Add & Norm
        mlp_out = hk.nets.MLP(
            [d * 4, d],
            activation=jax.nn.gelu
        )(x)
        
        # Add & Norm
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + mlp_out)
        
        return x

class TransformerNet(hk.Module):
    def __init__(self, num_actions, hidden_size, num_blocks, num_heads, seq_len):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size # Embed Dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.seq_len = seq_len

    def __call__(self, x, mask=None, is_training=False):
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
        
        if mask is not None:
             inf_mask = (1.0 - mask) * -1e9
             policy_logits = policy_logits + inf_mask

        # Value
        value = hk.Linear(2)(x)
        
        return policy_logits, value

class PrecomputedEmbedding(hk.Module):
    """Module to lookup vectors from precomputed feature tables"""
    def __init__(self, embedding_matrix, output_dim, name=None):
        super().__init__(name=name)
        self.embedding_matrix = embedding_matrix # shape: (TotalCards, FeatureDim)
        self.output_dim = output_dim

    def __call__(self, card_ids):
        # card_ids: (Batch, NumSlots) integer array
        
        # 1. Define the table as a constant parameter (not learned)
        table = hk.get_parameter(
            "feature_table",
            shape=self.embedding_matrix.shape,
            init=hk.initializers.Constant(self.embedding_matrix)
        )
        table = jax.lax.stop_gradient(table) # Fix it
        
        # 2. Lookup vectors corresponding to IDs
        # Negative IDs (-1.0) are treated as null (assuming index 0 is null, or clip)
        # Here, clip(0, num_cards-1) and then mask by valid ID is safe
        num_cards = self.embedding_matrix.shape[0]
        valid_mask = (card_ids >= 0) & (card_ids < num_cards)
        safe_ids = jnp.where(valid_mask, card_ids, 0)
        
        # x shape: (Batch, NumSlots, FeatureDim)
        x = jnp.take(table, safe_ids, axis=0)
        
        # Mask invalid IDs with zeros
        x = jnp.where(valid_mask[..., None], x, 0.0)
        
        # 3. Compress/transform to Transformer dimension (hidden_size)
        x = hk.Linear(self.output_dim)(x)
        return x

class HybridEmbedding(hk.Module):
    """Embedding combining general vectors (Static) and correction patches (Residual)"""
    def __init__(self, pretrained_matrix, output_dim, name=None):
        super().__init__(name=name)
        self.pretrained_matrix = pretrained_matrix # (NumCards, FeatureDim)
        self.output_dim = output_dim

    def __call__(self, card_ids):
        # 1. Static Path (text meaning)
        #    Linear projection is performed within PrecomputedEmbedding
        static_proj = PrecomputedEmbedding(
            self.pretrained_matrix, 
            self.output_dim, 
            name="static_path"
        )(card_ids)

        # 2. Residual Path (ID-specific exception correction)
        #    Initialized with w_init=0, absorbs exceptions as training progresses
        num_cards = self.pretrained_matrix.shape[0]
        valid_mask = (card_ids >= 0) & (card_ids < num_cards)
        safe_ids = jnp.where(valid_mask, card_ids, 0)

        residual_emb = hk.Embed(
            vocab_size=num_cards,
            embed_dim=self.output_dim,
            w_init=hk.initializers.Constant(0.0), 
            name="residual_path"
        )(safe_ids)

        # Mask invalid IDs with zeros
        residual_emb = jnp.where(valid_mask[..., None], residual_emb, 0.0)

        # 3. Add (Integration)
        return static_proj + residual_emb

class CardTransformerNet(hk.Module):
    def __init__(self, num_actions, embedding_matrix, hidden_size=64, num_blocks=2, num_heads=4):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_matrix = embedding_matrix
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

    def __call__(self, x, mask=None, is_training=False):
        # x: (Batch, ObsDim)
        batch_size = x.shape[0]
        
        # 1. Extract information
        turn_info = x[:, :39]
        board_part = x[:, 39:39+192].reshape(batch_size, 8, 24)
        hand_ids = x[:, 231:241].astype(jnp.int32)
        deck_ids = x[:, 241:261].astype(jnp.int32)
        opp_deck_count = x[:, 261:262]
        discard_ids = x[:, 262:272].astype(jnp.int32)
        opp_discard_ids = x[:, 272:282].astype(jnp.int32)
        
        # 2. Embedding (Board)
        board_card_ids = board_part[:, :, 11].astype(jnp.int32)
        board_card_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_board")(board_card_ids)
        
        # Board dynamic features
        indices = [i for i in range(24) if i != 11]
        board_features = board_part[:, :, indices]
        board_features = hk.Linear(self.hidden_size, name="lin_board_feat")(board_features)
        board_slot_repr = board_card_emb + board_features
        
        # 3. Embedding (Hand, Deck, Discard)
        hand_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_hand")(hand_ids)
        deck_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_deck")(deck_ids)
        discard_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_discard")(discard_ids)
        opp_discard_emb = HybridEmbedding(self.embedding_matrix, self.hidden_size, name="emb_opp_discard")(opp_discard_ids)

        # 4. Add Usability Flags (Concatenation)
        # Concatenate 4 flags to each token, resulting in hidden_size + 4 dimensional input
        if mask is not None:
            C = self.embedding_matrix.shape[0]
            # Offsets (match encoding.rs)
            off_atk = 1
            off_ret = 4
            off_abl = 8
            off_place = 12
            off_evolve = 12 + 4 * C
            off_play = 12 + 8 * C
            off_tool = 12 + 9 * C + 40
            
            # (1) Board flags: Attack 1, Attack 2, Ability, Retreat
            # My board (0..3)
            # Active
            my_atk1 = mask[:, off_atk:off_atk+1]
            my_atk2 = mask[:, off_atk+1:off_atk+2]
            my_ret = jnp.any(mask[:, off_ret:off_ret+4], axis=1, keepdims=True).astype(jnp.float32)
            # UseAbility
            my_abl = mask[:, off_abl:off_abl+4] # (B, 4)
            
            zero = jnp.zeros((batch_size, 1))
            # My slots (4, 4 flags)
            my_board_flags = jnp.stack([
                jnp.concatenate([my_atk1, my_atk2, my_abl[:, 0:1], my_ret], axis=1),
                jnp.concatenate([zero, zero, my_abl[:, 1:2], zero], axis=1),
                jnp.concatenate([zero, zero, my_abl[:, 2:3], zero], axis=1),
                jnp.concatenate([zero, zero, my_abl[:, 3:4], zero], axis=1),
            ], axis=1)
            # Opponent slots (all 0)
            opp_board_flags = jnp.zeros((batch_size, 4, 4))
            all_board_flags = jnp.concatenate([my_board_flags, opp_board_flags], axis=1)
            
            # (2) Hand flags: Usability (1st flag)
            batch_idx = jnp.arange(batch_size)[:, None]
            safe_hand_ids = jnp.where(hand_ids >= 0, hand_ids, 0)
            
            f_play = mask[batch_idx, off_play + safe_hand_ids]
            f_place = jnp.any(jnp.stack([mask[batch_idx, off_place + safe_hand_ids*4 + s] for s in range(4)], axis=-1), axis=-1)
            f_evolve = jnp.any(jnp.stack([mask[batch_idx, off_evolve + safe_hand_ids*4 + s] for s in range(4)], axis=-1), axis=-1)
            f_tool = jnp.any(jnp.stack([mask[batch_idx, off_tool + safe_hand_ids*4 + s] for s in range(4)], axis=-1), axis=-1)
            
            hand_usable = jnp.maximum(jnp.maximum(f_play, f_place), jnp.maximum(f_evolve, f_tool))
            hand_usable = jnp.where(hand_ids >= 0, hand_usable, 0.0)
            hand_flags = jnp.stack([hand_usable, jnp.zeros_like(hand_usable), jnp.zeros_like(hand_usable), jnp.zeros_like(hand_usable)], axis=-1)
            
            # (3) Others (Deck, Discard) are zero padded
            deck_flags = jnp.zeros((batch_size, 20, 4))
            discard_flags = jnp.zeros((batch_size, 10, 4))
            opp_discard_flags = jnp.zeros((batch_size, 10, 4))
            
            # Concatenate
            board_slot_repr = jnp.concatenate([board_slot_repr, all_board_flags], axis=-1)
            hand_emb = jnp.concatenate([hand_emb, hand_flags], axis=-1)
            deck_emb = jnp.concatenate([deck_emb, deck_flags], axis=-1)
            discard_emb = jnp.concatenate([discard_emb, discard_flags], axis=-1)
            opp_discard_emb = jnp.concatenate([opp_discard_emb, opp_discard_flags], axis=-1)
        else:
            # If mask is not provided (usually not expected), zero pad
            padding = jnp.zeros((batch_size, 1, 4))
            board_slot_repr = jnp.concatenate([board_slot_repr, jnp.tile(padding, (1, 8, 1))], axis=-1)
            hand_emb = jnp.concatenate([hand_emb, jnp.tile(padding, (1, 10, 1))], axis=-1)
            deck_emb = jnp.concatenate([deck_emb, jnp.tile(padding, (1, 20, 1))], axis=-1)
            discard_emb = jnp.concatenate([discard_emb, jnp.tile(padding, (1, 10, 1))], axis=-1)
            opp_discard_emb = jnp.concatenate([opp_discard_emb, jnp.tile(padding, (1, 10, 1))], axis=-1)

        # 5. Slot Positioning & Sequence Construction
        # Also adjust pos_emb to fit the concatenated dimensions (revert to hidden_size (256) for TPU v6e)
        
        # Project features (hidden_size + 4) to hidden_size to ensure 256 alignments
        projection_layer = hk.Linear(self.hidden_size, name="lin_feature_proj")
        
        board_slot_repr = projection_layer(board_slot_repr)
        hand_emb = projection_layer(hand_emb)
        deck_emb = projection_layer(deck_emb)
        discard_emb = projection_layer(discard_emb)
        opp_discard_emb = projection_layer(opp_discard_emb)

        def get_pos_emb(name, num_slots):
             # For TPU, fix hidden_size to 256
            return hk.get_parameter(name, [1, num_slots, self.hidden_size], init=hk.initializers.TruncatedNormal())
            
        board_pos = get_pos_emb("pos_board", 8)
        hand_pos = get_pos_emb("pos_hand", 10)
        deck_pos = get_pos_emb("pos_deck", 20)
        discard_pos = get_pos_emb("pos_discard", 10)
        opp_discard_pos = get_pos_emb("pos_opp_discard", 10)
        
        tokens = [
            board_slot_repr + board_pos,
            hand_emb + hand_pos,
            deck_emb + deck_pos,
            discard_emb + discard_pos,
            opp_discard_emb + opp_discard_pos
        ]
        x_seq = jnp.concatenate(tokens, axis=1) # (Batch, 58, hidden_size)
        
        # 6. Transformer Blocks
        for i in range(self.num_blocks):
            # TransformerBlock internally gets the input dimension hidden_size
            x_seq = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.hidden_size // self.num_heads,
                hidden_size=self.hidden_size,
                name=f"block_{i}"
            )(x_seq, is_training)
            
        # 7. Global Features Integration
        global_summary = jnp.mean(x_seq, axis=1)
        
        self_deck_count = jnp.sum(deck_ids >= 0, axis=1, keepdims=True).astype(jnp.float32)
        context_repr = jnp.concatenate([turn_info, self_deck_count, opp_deck_count], axis=-1)
        context_repr = hk.Linear(self.hidden_size, name="lin_context")(context_repr)
        context_repr = jax.nn.relu(context_repr)
        
        # final_repr is adjusted to match hidden_size
        final_repr = jnp.concatenate([global_summary, context_repr], axis=-1)
        final_repr = hk.Linear(self.hidden_size, name="lin_final")(final_repr)
        final_repr = jax.nn.relu(final_repr)
        
        # 8. Heads
        policy_logits = hk.Linear(self.num_actions, name="lin_policy")(final_repr)
        if mask is not None:
             inf_mask = (1.0 - mask) * -1e9
             policy_logits = policy_logits + inf_mask

        value = hk.Linear(2, name="lin_value")(final_repr)
        return policy_logits, value
