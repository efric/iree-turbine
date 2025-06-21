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
)
from .minimize_global_loads import (
    is_transposed_read,
    materialize_shape,
    update_write_dependencies,
)
from .global_to_shared_gathers import update_read_mapping_dynamic_values
from ..ops.wave_ops import Extract, Read, Write, Reshape
from ..wave.utils.classes import LDSTransposeRead
from ..wave.utils.run_utils import get_default_arch
import logging

logger = logging.getLogger(__name__)

"""
Optimize shared -> reg for transpose using lds.tr{n} intrinsics
TODO: extend support for more variants
"""


def is_transpose_read(node: fx.Node) -> bool:
    read = get_custom(node)
    if not isinstance(read, Read):
        return False

    src_shape = read.type.symbolic_shape
    if len(src_shape) <= 1:
        return False

    return is_transposed_read(read)


def feeds_mma_instruction(write: Write) -> bool:
    write_memory = write.memory

    for user_node in write_memory.users:
        custom_user = get_custom(user_node)
        if isinstance(custom_user, Read):
            for mma_user_node in user_node.users:
                mma_custom = get_custom(mma_user_node)

                if (
                    hasattr(mma_custom, "tkw_op_name")
                    and mma_custom.tkw_op_name == "mma"
                ):
                    return True

    return False


def meets_hw_transpose_requirements(
    read: Read, write: Write, constraints: list[Constraint]
):
    # breakpoint()
    if not get_default_arch() == "gfx950":
        return False

    write_memory = get_custom(write.memory)
    if write_memory.type.address_space != SHARED_ADDRESS_SPACE:
        return False

    if read.mapping_dynamic_vals:
        return False

    if read.type.dtype.bitwidth() != 8:
        return False

    # if not feeds_mma_instruction(write):
    #     return False

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }
    materialized_shape = materialize_shape(
        constraint_tile_size, read.type.symbolic_shape
    )
    breakpoint()
    # if any(s > 1 for s in materialized_shape[:-2]) or any(s <= 1 for s in materialized_shape[-2:]
    # ):
    #     logger.info(
    #         f"only last 2 dims transpose is supported, got {materialized_shape}"
    #     )
    #     return False

    # breakpoint()
    # if materialized_shape[-2] % 16 != 0 or materialized_shape[-1] % 8 != 0:
    #     return False

    hardware_constraint = get_hardware_constraint(constraints)
    if hardware_constraint.threads_per_wave < 16:
        return False

    return True

def transpose_last2(shape: Sequence[IndexSymbol]) -> list[IndexSymbol]:
    return list(shape[:-2]) + [shape[-1], shape[-2]]

def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint]
):
    """
    Mark shared memory allocations that can use hardware transpose.
    This is separate from in_thread_transpose optimization.
    """
    logger.info("mark_hardware_transpose_candidates")
    # breakpoint()
    candidates = trace.walk(is_transpose_read)
    breakpoint()

    rw_mem_seen = set()
    new_writes = defaultdict(list)
    new_reads = defaultdict(list)

    for read in candidates:
        read = get_custom(read)
        for write in read.users:
            if not isinstance(write, Write):
                continue

            if meets_hw_transpose_requirements(read, write, constraints):
                rw_mem = (read.memory, write.memory)
                if rw_mem not in rw_mem_seen:
                    rw_mem_seen.add(rw_mem)
                    mark_hw_transpose(write, new_writes, read, new_reads)

    for old_read, new_read in new_reads.items():
        new_read_fx_node = new_read[0] 

        for user in list(old_read.users.keys()):
              custom_user = get_custom(user)
              if isinstance(custom_user, Write):
                  # Find which argument index the old read is at
                  for i, arg in enumerate(user.args):
                      if arg == old_read:
                        custom_user.update_arg(i, new_read_fx_node)
    if new_writes:
        update_write_dependencies(new_writes, trace)


def mark_hw_transpose(write: Write, new_writes: dict, read: Read, new_reads):
    with write.graph.inserting_before(write.fx_node):
        dest = get_custom(write.memory)
        dest.update_arg("hardware_transpose", LDSTransposeRead.tr8_b64)
        # breakpoint()
        # current_shape = list(dest.distributed_shape)
        # current_shape[-1] = ((current_shape[-1] + 511) // 512) * 512
        # dest.update_arg("distributed_shape", tuple(current_shape))
        # dest.distributed_shape = tuple(current_shape)
        # breakpoint()
        hw_write = Write(
            write.register_,
            write.memory,
            write.elements_per_thread,
            mapping=write.mapping,
            mapping_dynamic_vals=write.mapping_dynamic_vals,
        ).add_to_graph(write.graph)

        hw_write.index = write.index
        new_writes[write.memory].append(hw_write)

        logger.info(f"Marked hardware transpose write: {hw_write}")

    mapping = read.mapping
    if read.mapping is not None:
        # this is just meant to give contiguous global loads
        src_shape = transpose_last2(read.type.symbolic_shape)
        out_mapping = {
            k: IndexMapping.iterator(i)
            for i, k in enumerate(src_shape)
        }
        # subs = {v: out_mapping[k] for k, v in mapping.output_mapping.items()}
        input_mapping = out_mapping.copy()
        # input_mapping = {
        #      k: safe_subs(v, subs, simultaneous=True)
        #      for k, v in mapping.input_mapping.items()
        # }
        mapping = IndexMapping(
            num_iterators=len(out_mapping),
            inputs=input_mapping,
            outputs=out_mapping,
            dynamic_val_mappings=mapping.dynamic_val_mappings
        )
        """
          (Pdb) mapping
IndexMapping(iters={$index0: 0, $index1: 1}, input_mapping={K: $index0, N: $index1}), output_mapping={K: $index0, N: $index1}, dynamic_val_mappings=()
(Pdb) read.mapping
IndexMapping(iters={$index0: 0, $index1: 1}, input_mapping={N: $index0, K: $index1}), output_mapping={N: $index0, K: $index1}, dynamic_val_mappings=()

otherwise you get 

          %54 = memref.load %reinterpret_cast[%c0] : memref<?xi8, strided<[1], offset: ?>>
          %55 = memref.load %reinterpret_cast[%c1280] : memref<?xi8, strided<[1], offset: ?>>
          %56 = memref.load %reinterpret_cast[%c2560] : memref<?xi8, strided<[1], offset: ?>>
          %57 = memref.load %reinterpret_cast[%c3840] : memref<?xi8, strided<[1], offset: ?>>
          %58 = memref.load %reinterpret_cast[%c5120] : memref<?xi8, strided<[1], offset: ?>>
          %59 = memref.load %reinterpret_cast[%c6400] : memref<?xi8, strided<[1], offset: ?>>
          %60 = memref.load %reinterpret_cast[%c7680] : memref<?xi8, strided<[1], offset: ?>>
          %61 = memref.load %reinterpret_cast[%c8960] : memref<?xi8, strided<[1], offset: ?>>
#         """
    with read.graph.inserting_before(read.fx_node):
                new_read = Read(
                    read.memory,
                    read.elements_per_thread,
                    mapping=mapping,
                    mapping_dynamic_vals=read.mapping_dynamic_vals,
                ).add_to_graph(read.graph)
                new_read.index = read.index
                new_read_custom = get_custom(new_read)
                new_read_custom.infer_type()
                if read.mapping_dynamic_vals:
                    update_read_mapping_dynamic_values(new_read_custom)
                # breakpoint()
                new_reads[read.fx_node].append(new_read)