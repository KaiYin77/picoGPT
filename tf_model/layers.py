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

        # Causal mask will be created in build
        self.causal_mask = None

    def build(self, input_shape):
        # Pre-compute causal mask (lower triangular matrix)
        # Shape: (block_size, block_size)
        # Use compute dtype to support mixed precision
        seq_len = self.block_size
        compute_dtype = self.compute_dtype if hasattr(self, 'compute_dtype') else tf.float32
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=compute_dtype), -1, 0)

        # Use add_weight with trainable=False instead of tf.Variable for better Keras 3 compatibility
        self.causal_mask = self.add_weight(
            name='causal_mask',
            shape=(seq_len, seq_len),
            dtype=compute_dtype,
            initializer=tf.constant_initializer(mask.numpy()),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x, training=False, k_cache=None, v_cache=None, cache_position=None):
        """
        Args:
            x: Input tensor of shape (B, T, C)
            training: Whether in training mode
            k_cache: Optional cached K tensor of shape (B, n_head, cache_len, head_dim)
            v_cache: Optional cached V tensor of shape (B, n_head, cache_len, head_dim)
            cache_position: Optional position offset for cache usage (scalar)

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

        # KV-Cache logic: if cache provided, use full cache for attention
        if k_cache is not None and v_cache is not None:
            # Use full caches for attention calculation
            k = k_cache
            v = v_cache
        else:
            # No cache, use new k and v
            k = k_new
            v = v_new


        # Scaled dot-product attention
        # att = (Q @ K^T) / sqrt(head_dim)
        # Use x.dtype to support mixed precision
        scale = tf.cast(head_dim, x.dtype)
        scale = tf.math.rsqrt(scale)

        # (B, n_head, T, T_total) where T_total = cache_len + T
        att = tf.matmul(q, k, transpose_b=True) * scale

        # Apply causal mask
        # Use only the relevant part of the pre-computed mask
        mask_value = tf.constant(-1e9, dtype=x.dtype)
        T_total = tf.shape(k)[2]  # Total key length including cache

        if cache_position is not None:
            # When using cache, mask based on cache_position
            # Query positions: [cache_position, cache_position + T)
            # Key positions: [0, cache_position + T)
            causal_mask_slice = self.causal_mask[cache_position:cache_position + T, :T_total]
        else:
            # Normal mode: use standard causal mask
            causal_mask_slice = self.causal_mask[:T, :T_total]

        att = tf.raw_ops.SelectV2(
            condition=causal_mask_slice == 0,
            t=mask_value,
            e=att,
        )

        # Softmax over the last dimension (key dimension)
        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, training=training)

        # Weighted sum of values: (B, n_head, T, head_dim)
        y = tf.matmul(att, v)

        # Reshape back to (B, T, C)
        y = tf.transpose(y, [0, 2, 1, 3])  # (B, T, n_head, head_dim)
        y = tf.reshape(y, [B, T, C])

        # Output projection with residual dropout
        y = self.c_proj(y)
        y = self.resid_dropout(y, training=training)

        # Return new k and v if cache is used
        if k_cache is not None and v_cache is not None:
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
        Args:
            x: Input tensor of shape (B, T, C)
            training: Whether in training mode
            k_cache: Optional cached K tensor for this block
            v_cache: Optional cached V tensor for this block
            cache_position: Optional position offset for cache usage

        Returns:
            If cache provided: (output, new_k_cache, new_v_cache)
            Otherwise: output tensor of shape (B, T, C)
        """
        # CIMv3 sequence: MHA → FFN → LN2
        # MHA with residual
        if k_cache is not None and v_cache is not None:
            attn_out, new_k, new_v = self.attn(x, training=training, k_cache=k_cache,
                                                v_cache=v_cache, cache_position=cache_position)
            x = x + attn_out
        else:
            x = x + self.attn(x, training=training)

        # FFN with residual, then layer norm
        x = self.ln_2(x + self.mlp(x, training=training))

        if k_cache is not None and v_cache is not None:
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
