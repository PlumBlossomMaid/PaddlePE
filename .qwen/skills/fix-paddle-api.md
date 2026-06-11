# paddlePE Skill — Fix PaddlePaddle API Compatibility

When migrating models across PaddlePaddle versions, common API changes to watch for.

```
/skill fix-paddle-api
```

## Common Issues

| Old API (2.x) | New API (3.x) |
|---------------|---------------|
| `paddle.hann_window()` | Manual compute: `0.5 * (1.0 - cos(2πn/(N-1)))` |
| `return_complex=True` | `paddle.signal.stft` always returns complex |
| `window="hann"` (string) | Pass precomputed `window=tensor` |
| `x.transpose([1,2])` | `perm = list(range(ndim)); perm[1], perm[2] = 2, 1; x.transpose(perm)` |
| `.cuda()` | `.to("gpu:0")` |
| `set_state_dict(..., assign=True)` | Not supported, remove `assign` arg |
| `paddle.std(unbiased=0)` | `unbiased=False` (bool, not int) |
| `paddle.zeros([1], place=...)` | `paddle.to_tensor([0.0]).to("gpu:0")` |
