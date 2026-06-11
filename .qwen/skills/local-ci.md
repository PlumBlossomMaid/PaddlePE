# paddlePE Skill — Local CI

Quick verification before committing.

```
/skill local-ci
```

## Steps

```bash
# 1. Ruff lint + format (Paddle style, ruff 0.15.0)
ruff check paddlepe/ tests/
ruff format paddlepe/ tests/ --check

# 2. Unit tests
python -m pytest tests/ -v --timeout=120

# 3. Import sanity
python -c "from paddlepe import PE; print('models:', PE.list_models())"

# 4. Remote server smoke test
python -c "
from paddlepe.remote import RemotePE
import numpy as np
pe = RemotePE(model='fcpe', auto_shutdown=True)
f0, conf = pe.infer(np.sin(np.linspace(0, 2*np.pi*440, 1600)).astype(np.float32), 16000)
print(f'F0 extracted: {len(f0)} frames')
pe.__del__()
"

# 5. CLI smoke test
paddlepe -l
python -c "from paddlepe.io import write_f0; import numpy as np; write_f0('/tmp/test.f0', np.ones(100, dtype=np.float32))"
python -c "from paddlepe.io import read_f0; f0, conf, sr, hop = read_f0('/tmp/test.f0'); print(f'Read {len(f0)} frames')"
```
