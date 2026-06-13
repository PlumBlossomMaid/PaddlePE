# paddlePE Skill — Add New PE Model

Add a new pitch extraction model to paddlePE.

```
/skill add-pe-model
```

## Steps

1. Create `paddlepe/models/<name>/` with:
   - `backbone.py` — model network (nn.Layer), with renamed classes
   - `infer.py` — `@PE.register("<name>")` class extending BasePE
   - `dataset.py` — dataset stub
   - `__init__.py`
   - (No `training.py` — training logic is in `scripts/train.py` via collators)

2. Register in `paddlepe/models/__init__.py`:
   ```python
   import paddlepe.models.<name>.infer  # noqa: F401
   ```

3. Add a collator in `paddlepe/training/collators/<name>.py`:
   ```python
   from paddlepe.training.collators.base import BaseCollator
   class <Name>Collator(BaseCollator):
       def __call__(self, batch):
           # Convert HDF5Dataset samples → model input + target
           return inp, target
   ```

4. Register in `paddlepe/training/collators/__init__.py`

5. Add to model registry in `scripts/train.py` `_model_registry()`

6. Add postproc defaults in `paddlepe/postproc/pipeline.py` `DEFAULT_CONFIG`

7. Add pretrained weights to `ckpts/<name>.pdparams`

8. Add tests in `tests/test_models.py`

9. Verify: `PE.create("<name>")` works
