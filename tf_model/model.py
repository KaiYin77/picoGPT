"""
PicoGPT model - TensorFlow implementation
Matches PyTorch architecture with weight tying and generation support
"""

import tensorflow as tf
from .layers import LayerNorm, Block, CustomEmbedding
from .config import PicoGPTConfig


class PicoGPT(tf.keras.Model):
    """
    PicoGPT: Tiny transformer model for CIMv3 hardware
    Features:
    - Weight tying between input embeddings and output projection
    - Post-norm transformer architecture
    - ReLU activation (CIMv3 constraint)
    - No bias in linear layers (CIMv3 preference)
    """

    def __init__(self, config: PicoGPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Token embeddings (wte) and positional embeddings (wpe)
        self.wte = CustomEmbedding(
            config.vocab_size,
            config.n_embd,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            name='wte'
        )

        self.wpe = CustomEmbedding(
            config.block_size,
            config.n_embd,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            name='wpe'
        )

        self.drop = tf.keras.layers.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = [
            Block(config, layer_idx=i, name=f'h_{i}')
            for i in range(config.n_layer)
        ]

        # Final layer norm
        self.ln_f = LayerNorm(name='ln_f')

        # Pre-compute positional indices to avoid tf.range
        self.pos_indices = tf.constant(list(range(self.config.block_size)), dtype=tf.int32, name='pos_indices')

        # Note: No separate lm_head layer
        # Output projection uses weight tying with wte.embeddings

    def call(self, idx, targets=None, training=False):
        """
        Forward pass through the model

        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices of shape (B, T) for loss computation (optional)
            training: Whether in training mode

        Returns:
            logits: Output logits of shape (B, T, vocab_size)
            loss: Scalar loss if targets provided, else None
        """
        # Get shape
        shape = tf.shape(idx)
        B = shape[0]
        T = shape[1]

        # Validate sequence length
        # tf.debugging.assert_less_equal(
        #     T, self.config.block_size,
        #     message=f"Sequence length cannot exceed block_size ({self.config.block_size})"
        # )

        # Token embeddings: (B, T, n_embd)
        tok_emb = self.wte(idx)

        # Position indices: (T,)
        pos = self.pos_indices[:T]

        # Positional embeddings: (T, n_embd)
        pos_emb = self.wpe(pos)

        # Combine embeddings: (B, T, n_embd)
        x = self.drop(tok_emb + pos_emb, training=training)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection with weight tying
        # Use embedding weights transposed as output projection
        # logits: (B, T, vocab_size)
        logits = tf.matmul(x, self.wte.embeddings, transpose_b=True)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Sparse categorical cross-entropy loss
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                targets,
                logits,
                from_logits=True
            )
            loss = tf.reduce_mean(loss)

        return logits, loss

    def call_with_cache(self, idx, k_caches, v_caches, cache_position, training=False):
        """
        Forward pass with KV-cache for efficient autoregressive generation

        Args:
            idx: Input token indices of shape (B, 1) - single new token
            k_caches: List of K cache tensors, one per layer, shape (B, n_head, cache_len, head_dim)
            v_caches: List of V cache tensors, one per layer, shape (B, n_head, cache_len, head_dim)
            cache_position: Current position in sequence (scalar int)
            training: Whether in training mode

        Returns:
            logits: Output logits of shape (B, 1, vocab_size)
            new_k_caches: Updated K cache list
            new_v_caches: Updated V cache list
        """
        # Get shape
        shape = tf.shape(idx)
        B = shape[0]
        T = shape[1]  # Should be 1 for single token generation

        # Token embeddings: (B, T, n_embd)
        tok_emb = self.wte(idx)

        # Positional embeddings: use cache_position as index
        # For single token (T=1), just gather the position embedding at cache_position
        # Reshape cache_position to be indexable
        pos_idx = tf.reshape(cache_position, [1])
        pos_emb = self.wpe(pos_idx)  # (1, n_embd)
        pos_emb = tf.expand_dims(pos_emb, 0)  # (1, 1, n_embd) to match tok_emb shape

        # Combine embeddings: (B, T, n_embd)
        x = self.drop(tok_emb + pos_emb, training=training)

        # Apply transformer blocks with cache
        new_k_caches = []
        new_v_caches = []
        for i, block in enumerate(self.blocks):
            x, new_k, new_v = block(x, training=training,
                                     k_cache=k_caches[i], v_cache=v_caches[i],
                                     cache_position=cache_position)
            new_k_caches.append(new_k)
            new_v_caches.append(new_v)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection with weight tying
        # logits: (B, T, vocab_size)
        logits = tf.matmul(x, self.wte.embeddings, transpose_b=True)

        return logits, new_k_caches, new_v_caches

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively

        Args:
            idx: Initial context of shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (None to disable)

        Returns:
            Generated sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if needed (only use last block_size tokens)
            T = tf.shape(idx)[1]
            idx_cond = idx if T <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass (no training, no targets)
            logits, _ = self(idx_cond, training=False)

            # Focus only on the last time step: (B, vocab_size)
            logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                top_k_val = tf.minimum(top_k, tf.shape(logits)[-1])
                values, _ = tf.nn.top_k(logits, k=top_k_val)
                min_value = values[:, -1:]

                # Set logits below top-k threshold to very negative value
                # Use logits.dtype for mixed precision compatibility
                logits = tf.raw_ops.SelectV2(
                    condition=logits < min_value,
                    t=tf.constant(-1e10, dtype=logits.dtype),
                    e=logits,
                )

            # Sample from the distribution using logits directly
            # tf.random.categorical expects log-probabilities (logits)
            idx_next = tf.random.categorical(
                logits,
                num_samples=1,
                dtype=tf.int32
            )  # (B, 1)

            # Append to sequence
            idx = tf.concat([idx, idx_next], axis=1)

        return idx

    def get_num_params(self, non_embedding=True):
        """
        Count the number of parameters in the model

        Args:
            non_embedding: If True, exclude positional embedding parameters

        Returns:
            Number of parameters
        """
        total = 0
        for var in self.trainable_variables:
            total += tf.size(var).numpy()

        # Subtract positional embedding if requested
        if non_embedding:
            total -= tf.size(self.wpe.embeddings).numpy()

        return total

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Configure AdamW optimizer
        Note: TensorFlow doesn't support per-parameter weight decay groups like PyTorch,
        so we use global weight decay

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters (beta1, beta2)

        Returns:
            Configured optimizer
        """
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=betas[0],
            beta_2=betas[1],
            epsilon=1e-8
        )

        return optimizer

    def get_config(self):
        """
        Serialize model configuration for Keras model saving

        Returns:
            Dictionary with serializable configuration
        """
        # Serialize PicoGPTConfig as a nested dictionary for robust saving
        config = super().get_config()
        config.update({
            'pico_gpt_config': self.config.__dict__
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Deserialize model from configuration

        Args:
            config: Dictionary with configuration

        Returns:
            PicoGPT model instance
        """
        from .config import PicoGPTConfig

        # Attempt to reconstruct PicoGPTConfig from the new, robust nested dictionary format
        if 'pico_gpt_config' in config:
            pico_gpt_config_dict = config['pico_gpt_config']
            model_config = PicoGPTConfig(**pico_gpt_config_dict)
        # Fallback for models saved with the old serialization format
        else:
            # Extract relevant keys to reconstruct PicoGPTConfig
            config_keys = [
                'block_size', 'vocab_size', 'n_layer',
                'n_head', 'n_embd', 'dropout', 'bias'
            ]
            model_config_dict = {key: config[key] for key in config_keys if key in config}
            model_config = PicoGPTConfig(**model_config_dict)

        return cls(model_config)
