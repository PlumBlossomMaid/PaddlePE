"""Model registry. Import model modules to trigger registration."""

# Import all model implementations to register them
import paddlepe.models.crepe.infer
import paddlepe.models.fcpe.infer
import paddlepe.models.penn.infer
import paddlepe.models.rmvpe.infer
import paddlepe.models.wrappers  # noqa: F401
