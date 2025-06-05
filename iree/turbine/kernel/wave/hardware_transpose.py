# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from typing import Sequence
from ..wave.constraints import (
    Constraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from ..lang.wave_types import IndexMapping
from ..ops.wave_ops import Read, Write, get_custom
from ..lang.global_symbols import *
from math import prod
import torch.fx as fx
from collections import defaultdict
from .utils.symbol_utils import safe_subs
from .utils.general_utils import (
    ceildiv,
    delinearize_index,
    get_hardware_constraint,
    remove_thread_indexing,
)
from .minimize_global_loads import (
    is_transposed_read,
    materialize_shape,
    update_write_dependencies,
)
from .global_to_shared_gathers import update_read_mapping_dynamic_values
from ..ops.wave_ops import Extract, Read, Write, Reshape
import logging

logger = logging.getLogger(__name__)


def is_transpose_read(node: fx.Node) -> bool:
    read = get_custom(node)
    if not isinstance(read, Read):
        return False

    src_shape = read.type.symbolic_shape
    if len(src_shape) <= 1:
        return False

    return is_transposed_read(read)


def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint]
):
    """
    Mark shared memory allocations that can use hardware transpose.
    This is separate from in_thread_transpose optimization.
    """
    logger.info("mark_hardware_transpose_candidates")

    #   hardware_constraint = get_hardware_constraint(constraints)

    #          logger.info("Hardware transpose not supported on this architecture")
    #          return

    # Find transpose read patterns
    candidates = trace.walk(is_transpose_read)
    #   breakpoint()

    # Build read-write pairs (similar to in_thread_transpose)
    mem_to_read_write = defaultdict(list)
    for read in candidates:
        read = get_custom(read)
        for write in read.users:
            if not isinstance(write, Write):
                continue
            mem_to_read_write[(read.memory, write.memory)].append((read, write))

    new_writes = defaultdict(list)
    for reads_writes in mem_to_read_write.values():
        read, write = reads_writes[0]
        alloc = get_custom(write.memory)
        replace_with_hardware_transpose_writes(read, write, new_writes, trace)
        logger.info("marked  in hardware transpose")
        breakpoint()

        # if should_mark_for_hardware_transpose(read, write, hardware_constraint):
        #    mark_allocation_for_hardware_transpose(write.memory, read, write)

    if new_writes:
        update_write_dependencies(new_writes, trace)


def replace_with_hardware_transpose_writes(
    read: Read, write: Write, new_writes: dict, trace: CapturedTrace
):
    with write.graph.inserting_before(write.fx_node):
        get_custom(write.memory).update_arg("transpose", True)
        hw_write = Write(
            write.register_,
            write.memory,
            write.elements_per_thread,
            mapping=write.mapping,
            mapping_dynamic_vals=write.mapping_dynamic_vals,
        ).add_to_graph(write.graph)

        hw_write.index = write.index

        # Collect in the same pattern as in_thread_transpose
        new_writes[write.memory].append(hw_write)

        logger.info(f"Created hardware transpose write: {hw_write}")
        breakpoint()
