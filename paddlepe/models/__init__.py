"""Model registry. Import model modules to trigger registration."""
# Import all model implementations to register them
import paddlepe.models.fcpe.infer  # noqa: F401
import paddlepe.models.rmvpe.infer  # noqa: F401
