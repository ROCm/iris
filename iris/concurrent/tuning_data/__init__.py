# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Committed tuning corpus + derived candidate sets for iris.concurrent.

See ``README.md`` for the data schema, methodology, and how to regenerate.

* ``collect_corpus.py`` -- on-device grid sweep -> ``data/<arch>_world<W>.json``
* ``derive_candidates.py`` -- corpus -> ``candidate_set.json`` (per-arch ordered configs)
* ``candidate_set.json`` -- the universally-good candidate set the tuner seeds from
"""

import json
import os

_HERE = os.path.dirname(__file__)


def load_candidate_set():
    """Return the committed per-arch candidate set (dict), or ``{}`` if absent."""
    path = os.path.join(_HERE, "candidate_set.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
