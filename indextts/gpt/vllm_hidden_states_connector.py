"""Windows-compatible hidden-state export connector for vLLM 0.24."""

import sys
import types

# vLLM's reference connector uses POSIX flock only for synchronizing its
# asynchronous file writer. IndexTTS waits until generation is finished before
# reading, and configures the connector with locking disabled. Provide a tiny
# import shim so the otherwise platform-neutral implementation loads on Windows.
if sys.platform == "win32" and "fcntl" not in sys.modules:
    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_SH = 1
    fcntl.LOCK_EX = 2
    fcntl.flock = lambda *_args, **_kwargs: None
    sys.modules["fcntl"] = fcntl

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E402
    ExampleHiddenStatesConnector,
)


class IndexTTSHiddenStatesConnector(ExampleHiddenStatesConnector):
    """Export cached GPT hidden states for consumption by s2mel."""

