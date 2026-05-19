import torch

def self_attention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )

    # Attention scores are dot products of inputs and inputs transposed
    attn_scores = inputs @ inputs.t()
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_scores)
    print(attn_weights)

    context = attn_weights @ inputs
    print(context)

if __name__ == "__main__":
    self_attention()