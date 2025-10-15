# Modified from Phil Wang's implementation: https://github.com/lucidrains/perceiver-pytorch

import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from math import pi
from torch import nn, einsum

def positional_encode(x):
    """
    Inputs:
        x: Raw input data of shape (B, ..., C). e.g., Image = (B, P, P, C)

    Outputs:
        axis_grid: Positional coordinates ranging from [-1, 1] of shape (..., d) e.g., Image = (224, 224, 2)
    """
    _, *axis, _ = x.shape
    axis_vec = []
    for dim in axis:
        axis_vec.append(torch.linspace(-1, 1, steps=dim, device=x.device, dtype=x.dtype))
    axis_grid = torch.stack(torch.meshgrid(*axis_vec, indexing='ij'), dim=-1)
    return axis_grid

def fourier_encode(x, max_freq=224, num_bands=64):
    """
    Inputs:
        x: Positional coordinates ranging from [-1, 1] of shape (..., d) e.g., Image = (224, 224, 2)
        max_freq: max frequency for encoding (mu in paper)
        num_bands: number of frequency bands to use (K in paper)

    Outputs:
        fourier_features: Encoded positions of shape (..., d * (2 * K + 1)) e.g., (224, 224, 261)
            d: each axis
                2 (sin + cosine) * K (number of frequency bands)
                + 1 (raw position)
    """
    device, dtype = x.device, x.dtype

    # Create frequency scales \f_k in paper: (K,)
    frequencies = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)

    # Add singleton dimensions to scales to make it broadcastable with x: e.g., (1, 1, 1, K)
    frequencies = rearrange(frequencies, "K -> " + " ".join("1" * x.ndim) + " K")

    # Add new dimension at end for broadcasting: (..., d, 1)
    x = x.unsqueeze(-1)

    # Scale positional coordinates by frequencies and π: (..., d, K)
    scaled_x = x * frequencies * pi

    # Compute sin/cos features: (..., d, 2*K)
    fourier_features = torch.cat([scaled_x.sin(), scaled_x.cos()], dim=-1)

    # Concatenate with original raw coordinates: [..., d, (2*K + 1)]
    fourier_features = torch.cat([fourier_features, x], dim=-1)

    # Flatten to create single feature axis: [..., d*(2*K + 1)]
    fourier_features = rearrange(fourier_features, "... n d -> ... (n d)")
    return fourier_features

class FeedForward(nn.Module):
    def __init__(self, dim, mult=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)
    
class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.norm = nn.LayerNorm(query_dim)
        self.to_qkv = nn.Linear(query_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=1, dim_head=261):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.norm_context = nn.LayerNorm(context_dim)
        self.norm = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        h = self.heads
        x = self.norm(x)
        context = self.norm_context(context)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class Perceiver(nn.Module):
    def __init__(
            self,
            max_freq,
            num_freq_bands,
            depth,
            input_channels,
            input_axis,
            num_latents,
            latent_dim,
            cross_heads,
            latent_heads,
            cross_dim_head,
            latent_dim_head,
            num_classes,
            self_per_cross_attn
   ):
        super().__init__()
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.input_axis = input_axis
        fourier_channels = input_axis * ((2 * num_freq_bands) + 1)
        context_dim = fourier_channels + input_channels
        # The latent array is randomly initialized using a truncated 
        # normal distribution with mean 0, standard deviation 0.02, 
        # and truncation bounds [-2, 2].
        self.latents = nn.Parameter((torch.randn(num_latents, latent_dim) * 0.02).clamp(-2., 2.))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            lat_attns = nn.ModuleList([])
            for _ in range(self_per_cross_attn):
                lat_attns.append(nn.ModuleList([
                    SelfAttention(latent_dim, latent_heads, latent_dim_head),
                    FeedForward(latent_dim)
                ]))
            self.layers.append(nn.ModuleList([
                CrossAttention(latent_dim, context_dim, cross_heads, cross_dim_head),
                FeedForward(latent_dim),
                lat_attns
            ]))
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )
    def forward(self, data):
        b, *axis, _ = data.shape
        assert len(axis) == self.input_axis
        pos = positional_encode(data)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)
        data = torch.cat([data, enc_pos], dim=-1)
        data = rearrange(data, 'b ... d -> b (...) d')
        x = repeat(self.latents, 'n d -> b n d', b=b)
        for cross_attn, cross_ff, latent_attns in self.layers:
            x = cross_attn(x, context=data) + x
            x = cross_ff(x) + x
            for lat_attn, lat_ff in latent_attns:
                x = lat_attn(x) + x
                x = lat_ff(x) + x
        return self.to_logits(x)

if __name__ == "__main__":

    torch.set_printoptions(precision=8)

    # ------------------------
    # Test Fourier Features
    # ------------------------

    axis = [224, 224]
    axis_pos = list(map(lambda size: torch.linspace(-1.0, 1.0, steps=size, dtype=torch.float32), axis))
    pos = torch.stack(torch.meshgrid(*axis_pos, indexing="ij"), dim=-1)
    enc_pos = fourier_encode(pos, max_freq=5, num_bands=6)
    expected = torch.tensor([8.74227766e-08, 8.09016824e-01, 9.51056480e-01, 3.09016943e-01, -5.87785602e-01, -1.00000000e00, -1.00000000e00, -5.87785423e-01])
    assert enc_pos.shape == (224, 224, 26), "Unexpected Fourier Features Shape"
    assert torch.allclose(expected, enc_pos[0, 0, :8]), "Unexpected Fourier Features Vaues"

    # ------------------------
    # Testing Attention
    # ------------------------

    torch.manual_seed(42)

    # Setup
    query_dim = 512  # D in paper
    context_dim = 29  # C in paper
    num_latents = 1024  # N in paper (N << M)
    num_pixels = 224 * 224  # M in paper

    x = torch.randn(size=(1, num_latents, query_dim))
    context = torch.randn(size=(1, num_pixels, context_dim))

    # Cross-attention: x: (b, M, d), context: (b, N, c) -> attn_out: (b, M, d)
    model = CrossAttention(query_dim, context_dim, heads=1, dim_head=64)
    attn_out = model(x, context)
    expected = torch.tensor([[0.08891644, 0.11354434, 0.06698097, 0.09210937, 0.05605839]])
    print(f"Cross Attention: {torch.allclose(attn_out[:, :5, 0], expected)}")
    print(f" Shape: {attn_out.shape == (1, 1024, 512)}")

    # Self-attention:: x: (b, M, d) -> attn_out: (b, M, d)
    model = SelfAttention(query_dim, heads=8, dim_head=64)
    attn_out = model(x)
    expected = torch.tensor([[-0.03783726, -0.04525265, -0.03628428, -0.04796932, -0.04929777]])
    print(f"Self Attention: {torch.allclose(attn_out[:, :5, 0], expected)}")
    print(f" Shape: {attn_out.shape == (1, 1024, 512)}")

    # ------------------------
    # Testing Perceiver
    # ------------------------

    torch.manual_seed(42)
    model = Perceiver(
        input_channels=3,
        input_axis=2,
        num_freq_bands=6,
        max_freq=10.0,
        depth=2,
        num_latents=256,
        latent_dim=512,
        cross_heads=2,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1000,
        self_per_cross_attn=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Defined Perceiver with: {total_params:,} params.")
    img = torch.randn(1, 224, 224, 3)  # 1 imagenet image
    output = model(img)  # (1, 1000)
    print(f"Data output {output.shape=}")
    expected1 = torch.tensor([[ 0.35286993,  0.20621315,  0.65644026,  0.16436583, -0.48988134]])
    expected2 = torch.tensor([[ 0.15717259,  1.01213932, -0.67046762, -0.22867519, -0.03702682]])
    output1 = output[:, 5:10]
    output2 = output[:, 65:70]
    print(f"Perceiver 1: {torch.allclose(output1, expected1)}")
    print(f"Perceiver 1: \n{output1=}, \n{expected1}")
    print(f"Perceiver 2: {torch.allclose(output2, expected2)}")
    print(f"Perceiver 2: \n{output2=}, \n{expected2}")