"""Compatibility shim — canonical code lives in iris.host.platform.utils"""
from iris.host.platform.utils import *  # noqa: F401,F403
from iris.host.platform.utils import do_bench, get_device_id_for_rank, is_simulation_env  # explicit
