import paddle
from einops import rearrange
from paddle import nn


def exists(val):
    return val is not None


class SinusoidalEmbeddings(nn.Layer):
    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (paddle.arange(0, dim, 2).astype(paddle.float32) / dim)
        )
        self.register_buffer('inv_freq', inv_freq)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), (
            'scale base must be defined if using xpos'
        )

        scale = (paddle.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent=False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device

        t = paddle.arange(seq_len).astype(self.inv_freq.dtype)
        freqs = paddle.einsum('i , j -> i j', t, self.inv_freq)
        freqs = paddle.concat((freqs, freqs), axis=-1)

        if not self.use_xpos:
            return freqs, paddle.ones([1])

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = paddle.concat((scale, scale), axis=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r=2)
    x1, x2 = x.unbind(dim=-2)
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale**-1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (
        rotate_half(k) * freqs.sin() * inv_scale
    )
    return q, k
