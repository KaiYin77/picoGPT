"""
Custom TensorFlow layers for PicoGPT
Matches PyTorch implementation with CIMv3 hardware constraints
"""

import tensorflow as tf
from .config import PicoGPTConfig


class LayerNorm(tf.keras.layers.Layer):
    """
    LayerNorm without bias for parameter efficiency
    Matches PyTorch: F.layer_norm(input, self.weight.shape, self.weight, None, 1e-5)
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        # Cast epsilon to input dtype for mixed precision support
        epsilon = tf.cast(self.epsilon, inputs.dtype)
        normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
        return self.gamma * normalized

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


class CausalSelfAttention(tf.keras.layers.Layer):
    """
    Causal self-attention with combined QKV projection
    Matches PyTorch CausalSelfAttention implementation
    """

    def __init__(self, config: PicoGPTConfig, **kwargs):
        super().__init__(**kwargs)
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Store config parameters for serialization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_rate = config.dropout
        self.block_size = config.block_size
        self.bias = config.bias

        # Combined QKV projection (3 * n_embd for Q, K, V)
        self.c_attn = tf.keras.layers.Dense(
            3 * self.n_embd,
            use_bias=self.bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            name='c_attn'
        )

        # Output projection
        self.c_proj = tf.keras.layers.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            name='c_proj'
        )

        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.resid_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Pre-compute causal and identity masks as constants to avoid tracking as weights
        self.causal_mask = tf.constant(
            tf.linalg.band_part(tf.ones((self.block_size, self.block_size)), -1, 0),
            dtype=tf.float32,
            name="causal_mask",
        )
        self.identity_mask = tf.constant(
            tf.eye(self.block_size, self.block_size),
            dtype=tf.float32,
            name="identity_mask",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False, k_cache=None, v_cache=None, cache_position=None):
        """
        Simplified call with minimal branching for TFLite compatibility.

        Args:
            x: Input tensor of shape (B, T, C)
            training: Whether in training mode
            k_cache: Cached K tensor of shape (B, n_head, cache_len, head_dim) or None
            v_cache: Cached V tensor of shape (B, n_head, cache_len, head_dim) or None
            cache_position: Position offset for cache usage (scalar) or None

        Returns:
            If cache provided: (output, new_k_cache, new_v_cache)
            Otherwise: output tensor of shape (B, T, C)
        """
        # Get shapes
        shape = tf.shape(x)
        B = shape[0]
        T = shape[1]
        C = self.n_embd
        head_dim = C // self.n_head

        # QKV projection: (B, T, 3*C)
        qkv = self.c_attn(x)

        # Split into Q, K, V: each (B, T, C)
        q, k, v = tf.split(qkv, 3, axis=-1)

        # Reshape to (B, T, n_head, head_dim) then transpose to (B, n_head, T, head_dim)
        q = tf.reshape(q, [B, T, self.n_head, head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])  # (B, n_head, T, head_dim)

        k_new = tf.reshape(k, [B, T, self.n_head, head_dim])
        k_new = tf.transpose(k_new, [0, 2, 1, 3])

        v_new = tf.reshape(v, [B, T, self.n_head, head_dim])
        v_new = tf.transpose(v_new, [0, 2, 1, 3])

        # Determine if using cache (evaluated at trace time, not runtime)
        use_cache = k_cache is not None and v_cache is not None

        # Select K, V for attention
        # IMPORTANT: Use explicit tf.identity to force TFLite to use the correct tensor
        # Python conditionals may not be properly resolved during TFLite conversion
        if use_cache:
            # Insert current token into cache for attention (so logits use tokens up to t)
            # Use a precomputed identity row to avoid OneHot/Select ops.
            pos = tf.cast(cache_position, tf.int32)
            exact_row = tf.gather(self.identity_mask, pos, axis=0)
            exact_mask = tf.reshape(exact_row, [1, 1, self.block_size, 1])
            exact_mask = tf.cast(exact_mask, x.dtype)
            k = k_cache + k_new * exact_mask
            v = v_cache + v_new * exact_mask
            k = tf.identity(k, name='k_from_cache_plus_new')
            v = tf.identity(v, name='v_from_cache_plus_new')
        else:
            k = tf.identity(k_new, name='k_from_new')
            v = tf.identity(v_new, name='v_from_new')

        # Scaled dot-product attention
        scale = tf.cast(head_dim, x.dtype)
        scale = tf.math.rsqrt(scale)
        att = tf.matmul(q, k, transpose_b=True) * scale

        # Apply causal mask (branch resolved at graph construction time)
        mask_value = tf.constant(-1e9, dtype=x.dtype)
        T_total = tf.shape(k)[2]

        # Compute mask based on whether we're using cache
        if cache_position is not None:
            # Window-based cache path: use position-aware masking (T=1 hardcoded to avoid StridedSlice)
            # For cache mode, T is always 1 (single token generation)
            # K/V caches are full block_size, so we use the full row (no column slicing needed!)
            mask_row = tf.minimum(cache_position, self.block_size - 1)
            # Extract single row from causal mask using tf.gather (avoids StridedSlice and Range ops)
            # Shape: (block_size,) - full row with all positions
            mask_row_data = tf.gather(self.causal_mask, mask_row, axis=0)
            mask_row_data = tf.cast(mask_row_data, x.dtype)
            # Expand to shape (1, block_size) to match attention weights shape
            causal_mask_slice = tf.expand_dims(mask_row_data, axis=0)
        else:
            # Standard path: use normal causal mask
            causal_mask_slice = self.causal_mask[:T, :T_total]
            causal_mask_slice = tf.cast(causal_mask_slice, x.dtype)

        # Apply mask without SelectV2 (mask is 1.0 for valid positions, 0.0 for masked)
        one = tf.constant(1.0, dtype=att.dtype)
        att = att * causal_mask_slice + (one - causal_mask_slice) * mask_value

        # Softmax and dropout
        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, training=training)

        # Weighted sum of values
        y = tf.matmul(att, v)

        # Reshape back to (B, T, C)
        y = tf.transpose(y, [0, 2, 1, 3])
        y = tf.reshape(y, [B, T, C])

        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y, training=training)

        # Return format (branch resolved at trace time)
        if use_cache:
            return y, k_new, v_new
        else:
            return y


class MLP(tf.keras.layers.Layer):
    """
    Compact MLP block with 2x expansion
    CIMv3 constraint: ReLU activation only, no bias
    """

    def __init__(self, config: PicoGPTConfig, **kwargs):
        super().__init__(**kwargs)

        # Store config parameters for serialization
        self.n_embd = config.n_embd
        self.bias = config.bias
        self.dropout_rate = config.dropout

        # 2x expansion for compact models (vs 4x in standard transformers)
        hidden_dim = 2 * self.n_embd

        self.c_fc = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=self.bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            name='c_fc'
        )

        self.c_proj = tf.keras.layers.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            name='c_proj'
        )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape (B, T, C)
            training: Whether in training mode

        Returns:
            Output tensor of shape (B, T, C)
        """
        x = self.c_fc(x)
        x = tf.nn.relu(x)  # CIMv3 only supports ReLU
        x = self.c_proj(x)
        x = self.dropout(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    """
    CIMv3-compatible transformer block
    Architecture: MHA → FFN → LN2 (post-norm only, no LN1)
    Residual connections around both attention and MLP
    """

    def __init__(self, config: PicoGPTConfig, layer_idx: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, name='attn')
        self.mlp = MLP(config, name='mlp')
        self.ln_2 = LayerNorm(name='ln_2')

    def call(self, x, training=False, k_cache=None, v_cache=None, cache_position=None):
        """
        Simplified call with minimal branching for TFLite compatibility.

        Args:
            x: Input tensor of shape (B, T, C)
            training: Whether in training mode
            k_cache: Cached K tensor for this block or None
            v_cache: Cached V tensor for this block or None
            cache_position: Position offset for cache usage or None

        Returns:
            If cache provided: (output, new_k_cache, new_v_cache)
            Otherwise: output tensor of shape (B, T, C)
        """
        # Determine if using cache (evaluated at trace time)
        use_cache = k_cache is not None and v_cache is not None

        # CIMv3 sequence: MHA → FFN → LN2
        # MHA with residual
        if use_cache:
            attn_out, new_k, new_v = self.attn(x, training=training, k_cache=k_cache,
                                                v_cache=v_cache, cache_position=cache_position)
            x = x + attn_out
        else:
            x = x + self.attn(x, training=training)

        # FFN with residual, then layer norm
        x = self.ln_2(x + self.mlp(x, training=training))

        # Return format (branch resolved at trace time)
        if use_cache:
            return x, new_k, new_v
        else:
            return x


class CustomEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform', **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs should be integer indices
        return tf.gather(self.embeddings, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['embeddings_initializer'] = tf.keras.initializers.deserialize(config['embeddings_initializer'])
        return cls(**config)
