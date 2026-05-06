# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    """
    Identifies a CSX in a cluster, along with its attached local host and the global host.

    Attributes:
        csx_id: The CSX identifier.
        localhost_id: The local host identifier.
        globalhost_id: The global host identifier.
    """

    id: int
    csx_id: int
    localhost_id: int
    globalhost_id: int


@dataclass
class Mesh:
    """
    Identifies a specific device mesh of boxes.

    Attributes:
        mesh_id: Unique identifier for the mesh.
        box_array: array of `Box` type elements.
    """

    mesh_id: int
    box_array: np.ndarray
