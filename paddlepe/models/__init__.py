"""Model registry. Import model modules to trigger registration."""

# Import all model implementations to register them
import paddlepe.models.fcpe.infer
import paddlepe.models.rmvpe.infer  # noqa: F401
