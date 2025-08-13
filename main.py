import torch
from foreteller import Foreteller


if __name__ == "__main__":
    # Test Foreseer
    embedding_dim = 128
    n_context = 4800
    num_rows = 5000
    num_cols = 10
    x = torch.randn(1, num_rows, num_cols, embedding_dim)

    config = {
        "d_in": embedding_dim,
        "d_out": 256,
        "num_heads": 8,
        "n_latents": 16,
        "n_isab_layers": 2,
        "n_mab_layers": 2,
        "dropout": 0.0,
        "num_labels": 5,
        "max_batch_size_col_sequence": 1024,
        "max_batch_size_row_sequence": 100,
    }
    model = Foreteller(config)
    out = model(x, n_context)
    print(f"Foreteller output shape: {out.shape}")
