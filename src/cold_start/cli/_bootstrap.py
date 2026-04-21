"""Side-effect imports that populate the component registries.

Importing this module triggers @register decorators in every concrete
component so the orchestrator can look them up by name.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

# models
from cold_start import models  # noqa: F401
from cold_start.models import mock_client as _mock  # noqa: F401

try:  # anthropic is optional at import time (fails without API key at construct)
    from cold_start.models import anthropic_client as _anth  # noqa: F401
except Exception:  # pragma: no cover
    pass

# task sources
from cold_start.tasks import toy as _toy  # noqa: F401
from cold_start.tasks import webarena as _wa  # noqa: F401
from cold_start.tasks import sanity as _sanity  # noqa: F401

# rewards
from cold_start.rewards import binary as _binary  # noqa: F401

# inference
from cold_start.inference import hedged_capital as _hc  # noqa: F401
from cold_start.inference import global_null as _gn  # noqa: F401

# policies
from cold_start.policies import uniform as _u  # noqa: F401
from cold_start.policies import epsilon_greedy as _eg  # noqa: F401
from cold_start.policies import thompson as _th  # noqa: F401
from cold_start.policies import spruce as _sp  # noqa: F401
