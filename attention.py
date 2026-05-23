import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out, gkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=gkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=gkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=gkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        attn_score = queries @ keys.T
        att_weights = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = att_weights @ values
        return context_vec

if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    sa = SelfAttention(d_in, d_out)
    context_vec = sa(inputs)
    print(context_vec)