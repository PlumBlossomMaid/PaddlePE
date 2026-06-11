# paddlePE Skill — Add New PE Model

Add a new pitch extraction model to paddlePE.

```
/skill add-pe-model
```

## Steps

1. Create `paddlepe/models/<name>/` with:
   - `backbone.py` — model network (nn.Layer), with renamed classes
   - `infer.py` — `@registry.register("<name>")` class extending BasePE
   - `training.py` — training stub
   - `dataset.py` — dataset stub
   - `__init__.py`

2. Register in `paddlepe/models/__init__.py`:
   ```python
   import paddlepe.models.<name>.infer  # noqa: F401
   ```

3. Add pretrained weights to `ckpts/<name>.pdparams`

4. Add tests in `tests/test_models.py`

5. Verify: `PE.create("<name>")` works
