"""
Tabular Transformer Implementation

This module implements a complete tabular transformer architecture with:
- Multi-head self-attention mechanisms
- Induced Set Attention Block (ISAB) for efficient processing of large sequences
- Two-dimensional attention for tabular data
- Configurable dropout and bias options
- Efficient einops-based tensor operations
- Memory-efficient chunked processing

Key Classes:
- SelfAttention: Core multi-head attention mechanism
- ISAB: Induced Set Attention Block for O(n) complexity
- TwoDimensionalAttention: 2D attention for tabular data
- AttentionBlock/ISABLayer: Wrapper classes with normalization and feed-forward
"""

import torch
import torch.nn as nn
import einops


# ============================================================================
# Utility Functions for Attention Masking
# ============================================================================

def build_attention_mask(num_rows: int, n_context: int) -> torch.Tensor:
    """
    Build attention mask for regular self-attention with context rows.
    
    Args:
        num_rows: Total number of rows in the sequence
        n_context: Number of context rows that can be attended to by all rows
        
    Returns:
        Tuple of (attention_mask, mask_matrix) where:
        - attention_mask: Mask for scaled_dot_product_attention (0 = attend, -inf = don't attend)
        - mask_matrix: Boolean mask matrix for debugging
    """
    context_attention_mask = torch.eye(num_rows)
    context_rows = torch.zeros(num_rows).bool()
    context_rows[:n_context] = True
    context_attention_mask[:, context_rows] = 1

    return (1.0 - context_attention_mask) * torch.finfo(context_attention_mask.dtype).min, context_attention_mask


def build_downsample_attention_mask(num_rows: int, n_context: int, n_latents: int = None) -> torch.Tensor:
    """
    Build attention mask for downsampling step in ISAB.
    
    Args:
        num_rows: Total number of rows in the sequence
        n_context: Number of context rows
        n_latents: Number of latent tokens
        
    Returns:
        Tuple of (attention_mask, mask_matrix) where:
        - attention_mask: Mask for scaled_dot_product_attention
        - mask_matrix: Boolean mask matrix for debugging
        
    The mask ensures:
    - Latents can only attend to context rows
    - Inference rows can attend to context rows and themselves
    """
    n_inference_rows = num_rows - n_context
    context_attention_mask = torch.zeros(n_latents + n_inference_rows, num_rows)
    context_attention_mask[-n_inference_rows:, -n_inference_rows:] = torch.eye(n_inference_rows)
    to_attend_to = torch.zeros(num_rows).bool()
    to_attend_to[:n_context] = True
    
    context_attention_mask[:, to_attend_to] = 1
    return (1.0 - context_attention_mask) * torch.finfo(context_attention_mask.dtype).min, context_attention_mask


def build_upsample_attention_mask(num_rows: int, n_context: int, n_latents: int = None) -> torch.Tensor:
    """
    Build attention mask for upsampling step in ISAB.
    
    Args:
        num_rows: Total number of rows in the sequence
        n_context: Number of context rows
        n_latents: Number of latent tokens
        
    Returns:
        Tuple of (attention_mask, mask_matrix) where:
        - attention_mask: Mask for scaled_dot_product_attention
        - mask_matrix: Boolean mask matrix for debugging
        
    The mask ensures:
    - Context rows can attend to latents
    - Inference rows can only attend to latents and themselves
    """
    n_inference_rows = num_rows - n_context
    context_attention_mask = torch.zeros(num_rows, n_latents + n_inference_rows)
    context_attention_mask[-n_inference_rows:, -n_inference_rows:] = torch.eye(n_inference_rows)
    to_attend_to = torch.zeros(n_latents + n_inference_rows).bool()
    to_attend_to[:n_latents] = True
    
    context_attention_mask[:, to_attend_to] = 1
    return (1.0 - context_attention_mask) * torch.finfo(context_attention_mask.dtype).min, context_attention_mask


# ============================================================================
# Core Attention Classes
# ============================================================================

class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism with support for different attention modes.
    
    This class implements a unified attention mechanism that can perform:
    - Regular self-attention between input and itself
    - Downsampling attention where latents attend to input
    - Upsampling attention where input attends to latents
    
    Attributes:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        d_out (int): Output dimension
        q (nn.Linear): Query projection layer
        kv (nn.Linear): Key-Value projection layer
        proj (nn.Linear): Output projection layer
        dropout (float): Dropout probability during training
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, 
                 dropout: float = 0.0, qkv_bias: bool = False):
        """
        Initialize the SelfAttention module.
        
        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability during training
            qkv_bias (bool): Whether to use bias in QKV projections
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
       
        self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.kv = nn.Linear(d_in, 2 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_in)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, latents: torch.Tensor = None, attn_mask: torch.Tensor = None, 
                n_context: int = None, attention_mode: str = "self") -> torch.Tensor:
        """
        Forward pass with unified attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_rows, d_in)
            latents (torch.Tensor, optional): Latent tensor for cross-attention modes
            attn_mask (torch.Tensor, optional): Attention mask for controlling attention patterns
            n_context (int, optional): Number of context rows for downsampling mode
            attention_mode (str): One of "self", "downsample", "upsample", or "auto"
                
        Returns:
            torch.Tensor: Output tensor of shape (batch, n_rows, d_in)
            
        Note:
            - "self": Regular self-attention between x and x
            - "downsample": Latents attend to x (for reduction to latents)
            - "upsample": x attends to latents (for upsampling from latents)
            - "auto": Automatically determines mode based on provided parameters
        """
        _, n_rows, _ = x.shape

        # Determine attention mode if not explicitly provided
        if attention_mode == "auto":
            if latents is not None:
                attention_mode = "downsample" if n_context is not None else "upsample"
            else:
                attention_mode = "self"
        
        # Route to appropriate attention method
        if attention_mode == "self":
            return self._self_attention(x, attn_mask)
        elif attention_mode == "downsample":
            return self._downsample_attention(x, latents, attn_mask, n_context)
        elif attention_mode == "upsample":
            return self._upsample_attention(x, latents, attn_mask)
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")

    def _self_attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Regular self-attention between x and x.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Self-attended output
        """
        q = self.q(x)
        k, v = self.kv(x).chunk(2, dim=-1)
        return self._apply_attention(q, k, v, attn_mask)

    def _downsample_attention(self, x: torch.Tensor, latents: torch.Tensor, 
                            attn_mask: torch.Tensor = None, n_context: int = None) -> torch.Tensor:
        """
        Downsample attention: latents attend to x.
        
        Args:
            x (torch.Tensor): Input tensor
            latents (torch.Tensor): Latent tokens (will be concatenated with inference rows)
            attn_mask (torch.Tensor, optional): Attention mask
            n_context (int): Number of context rows
            
        Returns:
            torch.Tensor: Contextualized latents
        """
        batch_size = x.shape[0]
        inference_rows = x[:, n_context:, :]
        # Expand latents to match batch size
        latents_expanded = einops.repeat(latents, "n_latents emb_dim -> b n_latents emb_dim", b=batch_size)
        latents_plus_inference_rows = torch.cat([latents_expanded, inference_rows], dim=1)
        q = self.q(latents_plus_inference_rows)
        k, v = self.kv(x).chunk(2, dim=-1)
        return self._apply_attention(q, k, v, attn_mask)

    def _upsample_attention(self, x: torch.Tensor, latents: torch.Tensor, 
                          attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Upsample attention: x attends to latents.
        
        Args:
            x (torch.Tensor): Input tensor
            latents (torch.Tensor): Contextualized latent tokens
            attn_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Upsampled output
        """
        q = self.q(x)
        k, v = self.kv(latents).chunk(2, dim=-1)
        return self._apply_attention(q, k, v, attn_mask)

    def _apply_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                        attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply the core attention mechanism.
        
        Args:
            q (torch.Tensor): Query tensor
            k (torch.Tensor): Key tensor
            v (torch.Tensor): Value tensor
            attn_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Attention output
        """
        # Reshape for multi-head attention
        q, k, v = map(lambda x: einops.rearrange(x, "b n_rows (n_heads head_dim) -> b n_heads n_rows head_dim", 
                                                 n_heads=self.num_heads, head_dim=self.head_dim), 
                      [q, k, v])
        # Apply dropout only during training
        use_dropout = 0. if not self.training else self.dropout

        # Compute scaled dot-product attention
        context_vec = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=False)
        
        # Combine multiple attention heads back into a single tensor
        context_vec = einops.rearrange(context_vec, "b n_heads n_rows head_dim -> b n_rows (n_heads head_dim)")

        # Apply final output projection
        context_vec = self.proj(context_vec)

        return context_vec


class InducedSetAttention(nn.Module):
    """
    Induced Set Attention for efficient processing of large sequences.
    
    This class reduces computational complexity by first downsampling to a small set of
    learned latent tokens, then upsampling back to the original sequence length.
    This allows processing of long sequences with O(n) complexity instead of O(n²).
    
    The architecture consists of two attention steps:
    1. Downsampling: Latents attend to input with proper masking
    2. Upsampling: Input attends to contextualized latents with causal masking
    
    Attributes:
        attn1 (SelfAttention): Downsampling attention layer
        attn2 (SelfAttention): Upsampling attention layer
        n_latents (int): Number of latent tokens
        latents (nn.Parameter): Learnable latent tokens
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, n_latents: int,
                 dropout: float = 0.0, qkv_bias: bool = False):
        """
        Initialize the ISAB module.
        
        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension
            num_heads (int): Number of attention heads
            n_latents (int): Number of latent tokens (typically much smaller than sequence length)
            dropout (float): Dropout probability during training
            qkv_bias (bool): Whether to use bias in QKV projections
        """
        super().__init__()

        self.attn1 = SelfAttention(d_in, d_out, num_heads, dropout, qkv_bias)
        self.attn2 = SelfAttention(d_in, d_out, num_heads, dropout, qkv_bias)
        self.n_latents = n_latents
        self.latents = nn.Parameter(torch.randn(n_latents, d_in))

    def forward(self, x: torch.Tensor, n_context: int) -> torch.Tensor:
        """
        Forward pass through the ISAB module.
        
        Args:
            x (torch.Tensor): Input tensor of shape
            n_context (int): Number of context rows that can be attended to
            
        Returns:
            torch.Tensor: Output tensor of shape
            
        The forward pass consists of two steps:
        1. Downsampling: Latents attend to input with masking to prevent information leakage
        2. Upsampling: Input attends to contextualized latents with causal masking
        """
        # First step: downsample to latents
        downsample_mask, _ = build_downsample_attention_mask(num_rows=x.shape[1], n_context=n_context, n_latents=self.n_latents)
        latents = self.attn1(x, self.latents, attn_mask=downsample_mask, n_context=n_context, attention_mode="downsample")

        # Second step: upsample from latents to output
        upsample_mask, _ = build_upsample_attention_mask(num_rows=x.shape[1], n_context=n_context, n_latents=self.n_latents)
        x = self.attn2(x, latents, attn_mask=upsample_mask, attention_mode="upsample")
        return x

class Downsampler(nn.Module):
    """
    Downsampler that compresses 4D tabular data to latent tokens using induced set attention.
    
    This class processes tabular data of shape (B, R, C, D) by:
    1. Rearranging to (B*C, R, D) for column-wise processing
    2. Applying downsampling attention to compress rows to latents
    3. Rearranging back to (B, n_latents, C, D) for further processing
    
    Attributes:
        attn (SelfAttention): Core attention mechanism for downsampling
        n_latents (int): Number of latent tokens
        latents (nn.Parameter): Learnable latent tokens
    """
    def __init__(self, d_in, d_out, num_heads, n_latents, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.attn = SelfAttention(d_in, d_out, num_heads, dropout, qkv_bias)
        self.n_latents = n_latents
        self.latents = nn.Parameter(torch.randn(n_latents, d_in))
    
    def forward(self, x, n_context):
        # Input x is (B, R, C, D) - rearrange to (B*C, R, D) for column-wise processing
        b, r, c, d = x.shape
        x = einops.rearrange(x, "b r c d -> (b c) r d")
        
        # Downsample to (B*C, n_latents, D)
        downsample_mask, _ = build_downsample_attention_mask(
            num_rows=x.shape[1], n_context=n_context, n_latents=self.n_latents
        )
        
        x = self.attn(x, self.latents, attn_mask=downsample_mask, 
                        n_context=n_context, attention_mode="downsample")
        
        # Rearrange back to (B, n_latents, C, D)
        x = einops.rearrange(x, "(b c) n_latents d -> b n_latents c d", b=b)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention Block (MAB) that wraps SelfAttention with proper masking.
    
    This class provides a clean interface for using SelfAttention with context-aware masking
    for tabular data processing.
    
    Attributes:
        attn (SelfAttention): Core self-attention mechanism
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, dropout: float = 0.0, qkv_bias: bool = False):
        """
        Initialize the MAB module.
        
        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability during training
            qkv_bias (bool): Whether to use bias in QKV projections
        """
        super().__init__()
        self.attn = SelfAttention(d_in, d_out, num_heads, dropout, qkv_bias)

    def forward(self, x: torch.Tensor, n_context: int = None) -> torch.Tensor:
        """
        Forward pass with optional context masking.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_rows, d_in)
            n_context (int, optional): Number of context rows for masking
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, n_rows, d_in)
        """
        if n_context is not None:
            attn_mask, _ = build_attention_mask(num_rows=x.shape[1], n_context=n_context)
            return self.attn(x, attn_mask=attn_mask, attention_mode="self")
        else:
            # For column-wise attention, no mask needed (every column can attend to every other column)
            return self.attn(x, attention_mode="self")


# ============================================================================
# Complete Layers with Normalization and Feed-Forward
# ============================================================================

class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation and residual connections.
    """
    def __init__(self, d_in, mult=2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_in * mult, bias=False)
        self.fc2 = nn.Linear(d_in, d_in * mult, bias=False)
        self.fc3 = nn.Linear(d_in * mult, d_in, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class MAB(nn.Module):
    """
    Multi-head attention layer with normalization and feed-forward network.
    """
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_in, d_out, num_heads, dropout, qkv_bias)
        self.ff = FeedForward(d_in)
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)

    def forward(self, x, n_context=None):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, n_context=n_context)
        x = x + shortcut
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


class ISAB(nn.Module):
    """
    Induced Set Attention layer with normalization and feed-forward network.
    """
    def __init__(self, d_in, d_out, num_heads, n_latents, dropout=0.0):
        super().__init__()
        self.attn = InducedSetAttention(d_in, d_out, num_heads, n_latents, dropout)
        self.ff = FeedForward(d_in)
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)

    def forward(self, x, n_context=None):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, n_context=n_context)
        x = x + shortcut
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


# ============================================================================
# Two-Dimensional Attention for Tabular Data
# ============================================================================

class TwoDimensionalISAB(nn.Module):
    """
    Two-dimensional attention mechanism for tabular data processing using ISAB for row attention.
    
    This module applies attention in two sequential passes:
    1. Column attention: Each row attends to all columns (full attention, no masking)
    2. Row attention: Each column attends to rows with optional context masking using ISAB
    
    The module supports both fast mode (processes all data at once) and chunked mode
    (processes in smaller batches to manage memory usage).
    
    Memory Management:
    - Column attention processes (B*R, C, D) sequences where R >> C typically
    - Row attention processes (B*C, R, D) sequences where R >> C typically
    - Since rows >> columns, the pseudo-batch size for row attention must be much smaller
      to prevent OOM errors. The default max_batch_size_row_sequence=100 reflects this.
    
    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension  
        num_heads (int): Number of attention heads
        n_latents (int): Number of latent tokens for ISAB row attention
        dropout (float, optional): Dropout probability (default: 0.0)
        max_batch_size_col_sequence (int, optional): Maximum batch size for column attention chunks (default: 1024)
        max_batch_size_row_sequence (int, optional): Maximum batch size for row attention chunks (default: 100)
                                          Should be much smaller than col_sequence due to R >> C
    """
    def __init__(self, d_in, d_out, num_heads, n_latents, dropout=0.0, 
                 max_batch_size_col_sequence=1024, max_batch_size_row_sequence=100):
        super().__init__()
        
        self.across_row_attention = ISAB(d_in, d_out, num_heads, n_latents, dropout)
        self.across_column_attention = MAB(d_in, d_out, num_heads, dropout)
        self.max_batch_size_col_sequence = max_batch_size_col_sequence
        self.max_batch_size_row_sequence = max_batch_size_row_sequence

    def fast_forward(self, x, n_context):
        """
        Fast forward pass without chunking - processes all data at once.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, R, C, D)
            n_context (int): Number of context rows for row attention masking
            
        Returns:
            torch.Tensor: Output tensor of shape (B, R, C, D)
        """
        assert x.dim() == 4, "Expected input of shape (B, R, C, D)"
        batch_size, num_rows, num_cols, dim = x.shape

        # 1) Column attention: treat each row as an item in the batch
        cols_as_sequence_in = einops.rearrange(x, "b r c d -> (b r) c d")
        cols_as_sequence_out = self.across_column_attention(cols_as_sequence_in, n_context=None)
        x = einops.rearrange(cols_as_sequence_out, "(b r) c d -> b r c d", b=batch_size)

        # 2) Row attention: treat each column as an item in the batch
        rows_as_sequence_in = einops.rearrange(x, "b r c d -> (b c) r d")
        rows_as_sequence_out = self.across_row_attention(rows_as_sequence_in, n_context=n_context)
        x = einops.rearrange(rows_as_sequence_out, "(b c) r d -> b r c d", b=batch_size)
        return x
    
    def forward(self, x, n_context, fast_mode=True):
        """
        Forward pass with optional chunking for memory management.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, R, C, D)
            n_context (int): Number of context rows for row attention masking
            fast_mode (bool): If True, processes all data at once. If False, uses chunking.
            
        Returns:
            torch.Tensor: Output tensor of shape (B, R, C, D)
            
        Note:
            When fast_mode=False, the module processes data in chunks to manage memory:
            - Column attention: chunks of size max_batch_size_col_sequence over (B*R, C, D)
            - Row attention: chunks of size max_batch_size_row_sequence over (B*C, R, D)
            
            Since R >> C typically, row attention uses much smaller chunk sizes to prevent OOM.
        """
        if fast_mode:
            return self.fast_forward(x, n_context)
        
        assert x.dim() == 4, "Expected input of shape (B, R, C, D)"
        batch_size, num_rows, num_cols, dim = x.shape

        # 1) Column attention: chunked processing for memory efficiency
        cols_as_sequence_in = einops.rearrange(x, "b r c d -> (b r) c d")
        pseudo_batch_size = cols_as_sequence_in.shape[0]
        
        out_flat = torch.empty_like(cols_as_sequence_in)
        chunk_size = min(self.max_batch_size_col_sequence, pseudo_batch_size)
        for i in range(0, pseudo_batch_size, chunk_size):
            chunk = cols_as_sequence_in[i:i + chunk_size]
            out_flat[i:i + chunk_size] = self.across_column_attention(chunk, n_context=None)

        x = einops.rearrange(out_flat, "(b r) c d -> b r c d", b=batch_size)

        # 2) Row attention: chunked processing with much smaller batch sizes
        # Since R >> C, we need smaller chunks to prevent OOM
        rows_as_sequence_in = einops.rearrange(x, "b r c d -> (b c) r d")
        pseudo_batch_size = rows_as_sequence_in.shape[0]
        
        out_flat = torch.empty_like(rows_as_sequence_in)
        chunk_size = min(self.max_batch_size_row_sequence, pseudo_batch_size)
        for i in range(0, pseudo_batch_size, chunk_size):
            chunk = rows_as_sequence_in[i:i + chunk_size]
            out_flat[i:i + chunk_size] = self.across_row_attention(chunk, n_context=n_context)

        x = einops.rearrange(out_flat, "(b c) r d -> b r c d", b=batch_size)

        return x


class TwoDimensionalMAB(nn.Module):
    """
    Two-dimensional attention mechanism for tabular data processing using MAB for both row and column attention.
    
    This module applies attention in two sequential passes:
    1. Column attention: Each row attends to all columns using MAB
    2. Row attention: Each column attends to rows using MAB with optional context masking
    
    This approach provides consistent attention mechanisms for both dimensions.
    
    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension  
        num_heads (int): Number of attention heads
        dropout (float, optional): Dropout probability (default: 0.0)
    """
    def __init__(self, d_in, d_out, num_heads, dropout=0.0):
        super().__init__()
        
        # Both row and column attention use MAB
        self.across_row_attention = MAB(d_in, d_out, num_heads, dropout)
        self.across_column_attention = MAB(d_in, d_out, num_heads, dropout)

    def forward(self, x, n_context):
        """
        Fast forward pass without chunking - processes all data at once.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, R, C, D)
            n_context (int): Number of context rows for row attention masking
            
        Returns:
            torch.Tensor: Output tensor of shape (B, R, C, D)
        """
        assert x.dim() == 4, "Expected input of shape (B, R, C, D)"
        batch_size, num_rows, num_cols, dim = x.shape

        # 1) Column attention: treat each row as an item in the batch
        cols_as_sequence_in = einops.rearrange(x, "b r c d -> (b r) c d")
        cols_as_sequence_out = self.across_column_attention(cols_as_sequence_in, n_context=None)
        x = einops.rearrange(cols_as_sequence_out, "(b r) c d -> b r c d", b=batch_size)

        # 2) Row attention: treat each column as an item in the batch
        rows_as_sequence_in = einops.rearrange(x, "b r c d -> (b c) r d")
        rows_as_sequence_out = self.across_row_attention(rows_as_sequence_in, n_context=n_context)
        x = einops.rearrange(rows_as_sequence_out, "(b c) r d -> b r c d", b=batch_size)
        return x


class TableEncoder(nn.Module):
    """
    Tabular Transformer model.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.isab_blocks = nn.ModuleList(
            [TwoDimensionalISAB(config["d_in"], config["d_out"], config["num_heads"], config["n_latents"], config["dropout"], 
                                max_batch_size_col_sequence=config.get('max_batch_size_col_sequence', 1024),
                                max_batch_size_row_sequence=config.get('max_batch_size_row_sequence', 100)) for _ in range(config["n_isab_layers"])])
        
        self.downsampler = Downsampler(config["d_in"], config["d_out"], config["num_heads"], config["n_latents"], config["dropout"])

        self.mab_blocks = nn.ModuleList(
            [TwoDimensionalMAB(config["d_in"], config["d_out"], config["num_heads"], config["dropout"]) for _ in range(config["n_mab_layers"])])
        
        self.final_norm = nn.LayerNorm(config["d_in"])

    def forward(self, x, n_context):
        _, total_rows, _, _ = x.shape
        # Process through ISAB blocks
        for block in self.isab_blocks:
            x = block(x, n_context)

        # Downsample to latents
        x = self.downsampler(x, n_context)
        
        # Process through MAB blocks
        for block in self.mab_blocks:
            x = block(x, n_context)

        x = self.final_norm(x)
        n_inference_rows = total_rows - n_context
        # extract n_inference_rows and target column
        to_be_predicted = x[:, -n_inference_rows:, -1, :]
        
        return to_be_predicted
