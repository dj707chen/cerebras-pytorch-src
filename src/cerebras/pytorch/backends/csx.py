"""CSX performance/debug flag containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union


@dataclass
class _Precision:
    optimization_level: int = 1


@dataclass
class _Performance:
    micro_batch_size: Union[None, int, Literal["auto", "explore"], Dict[str, Dict[str, int]]] = "auto"
    transfer_processes: int = 5
    use_speculative_optimizers: bool = True


@dataclass
class _Debug:
    retrace_every_iteration: bool = False
    lazy_initialization: bool = True
    debug_args: Any = None
    ini: Any = None
    compile_crd_memory_gi: Optional[int] = None
    execute_crd_memory_gi: Optional[int] = None
    wrk_memory_gi: Optional[int] = None
    act_memory_gi: Optional[int] = None
    cmd_memory_gi: Optional[int] = None
    wgt_memory_gi: Optional[int] = None


precision = _Precision()
performance = _Performance()
debug = _Debug()
