"""Compatibility shim — IrisGluon is now just Iris, device context in iris.device.gluon"""
import warnings
warnings.warn(
    "iris.experimental.iris_gluon is deprecated. "
    "Use iris.iris() for host class and iris.device.gluon.context.IrisDeviceCtx for device context.",
    DeprecationWarning,
    stacklevel=2,
)
from iris.device.gluon.context import IrisDeviceCtx  # noqa: F401
from iris.device.gluon.tracing import GluonDeviceTracing  # noqa: F401
from iris.host.iris import Iris as IrisGluon  # noqa: F401
from iris.host.iris import iris  # noqa: F401
