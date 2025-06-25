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
from .utils.symbol_utils import safe_subs, subs_idxc
from .utils.general_utils import (
    ceildiv,
    delinearize_index,
    get_hardware_constraint,
)
from .minimize_global_loads import (
    is_transposed_read,
    materialize_shape,
    update_write_dependencies,
    calculate_hardware_transpose_coverage,
    construct_hw_transpose_access_pattern,
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

def combine_index(
    index1: dict[IndexSymbol, IndexSequence],
    index2: dict[IndexSymbol, IndexSequence],
    fastest_dim: IndexSymbol,
    fastest_dim_vec_size: int,
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes two index sequences and combines them.
    """
    assert len(index1) == len(index2)
    return {
        key: IndexSequence(
            index1[key].start + index2[key].start,
            fastest_dim_vec_size if key == fastest_dim else 1,
            1,
        )
        for key in index2
    }

def remove_thread_indexing(
    index: dict[IndexSymbol, IndexSequence]
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes the index sequence for a global read and removes all
    thread level indexing.
    """
    subs = {t: 0 for t in [THREAD_0, THREAD_1, THREAD_2, GPR_NUM]}
    return {key: safe_subs(index[key], subs) for key in index}

def create_generic_hardware_transpose_index(
    original_index: dict, read: Read, constraints: list[Constraint]
) -> dict:
    """
    Generic hardware transpose index calculation
    """
    hardware_constraint = get_hardware_constraint(constraints)
    constraint_tile_size = {
        c.dim: c.tile_size for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }

    effective_shape = transpose_last2(read.type.symbolic_shape)
    materialized_shape = materialize_shape(constraint_tile_size, effective_shape)

    # Get concrete values using subs_idxc (like in_thread_transpose.py does)
    tile_rows = subs_idxc(materialized_shape[-2])  # 32 for Matrix B
    tile_cols = subs_idxc(materialized_shape[-1])  # 64 for Matrix B
    elements_per_thread = subs_idxc(read.elements_per_thread)  # 8

    # FIX: Check for THREAD_1 correctly using free_symbols
    has_thread_y = False
    for index_seq in original_index.values():
        if THREAD_1 in index_seq.start.free_symbols:
            has_thread_y = True
            break

    logger.info(f"Hardware transpose: {tile_rows}x{tile_cols}, has_thread_y={has_thread_y}")

    # Extract global offsets (like in_thread_transpose.py:203)
    global_index = remove_thread_indexing(original_index)

    # Create new thread-local index
    new_thread_index = {}
    last_two_dims = list(effective_shape[-2:])

    if has_thread_y:
        # Multi-wave case: Need to distribute 128x2=256 threads across 32x64 matrix
        # Each thread loads 8 elements, so capacity = 256*8 = 2048 elements
        # Matrix B = 32*64 = 2048 elements (perfect fit)

        # Pattern that should generate: (s0 floordiv 4) mod 32 for rows
        thread_row = sympy.Mod(sympy.floor(THREAD_0 / 4), tile_rows)

        # Pattern for columns using both thread dimensions
        thread_col = THREAD_1 * (tile_cols // 2) + sympy.Mod(THREAD_0, 4) * elements_per_thread

    else:
        # Single wave case (original working 32x16)
        linear_id = hardware_constraint.linearized_thread_id
        col_groups = ceildiv(tile_cols, elements_per_thread)
        threads_per_col_group = hardware_constraint.threads_per_wave // col_groups

        thread_row = sympy.Mod(linear_id, threads_per_col_group)
        col_group = sympy.floor(linear_id / threads_per_col_group)
        thread_col = col_group * elements_per_thread

    new_thread_index[last_two_dims[0]] = IndexSequence(thread_row, 1, 1)
    new_thread_index[last_two_dims[1]] = IndexSequence(thread_col, elements_per_thread, 1)

    # Combine with global offset (like in_thread_transpose.py:231-236)
    final_index = combine_index(
        global_index,
        new_thread_index,
        last_two_dims[1],
        elements_per_thread
    )

    # Preserve other dimensions
    for dim, index_seq in original_index.items():
        if dim not in final_index:
            final_index[dim] = index_seq

    return final_index

def create_generic_hardware_transpose_index(
    original_index: dict, read: Read, constraints: list[Constraint]
) -> dict:
    """
    Generic hardware transpose index - handles workgroup tiles, not full matrices
    """
    # Extract problem parameters
    hardware_constraint = get_hardware_constraint(constraints)
    constraint_tile_size = {
        c.dim: c.tile_size for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }

    # Get effective matrix shape (after transpose_last2 in mark_hw_transpose)
    effective_shape = transpose_last2(read.type.symbolic_shape)

    # IMPORTANT: This gives us the WORKGROUP TILE size, not the full matrix!
    # For a 1024x1024 matrix with BLOCK_M=32, BLOCK_N=16, this returns [32, 16]
    materialized_shape = materialize_shape(constraint_tile_size, effective_shape)

    # Extract concrete tile dimensions
    tile_rows = subs_idxc(materialized_shape[-2]) if hasattr(materialized_shape[-2], 'subs') else materialized_shape[-2]
    tile_cols = subs_idxc(materialized_shape[-1]) if hasattr(materialized_shape[-1], 'subs') else materialized_shape[-1]
    elements_per_thread = subs_idxc(read.elements_per_thread)
    threads_per_wave = hardware_constraint.threads_per_wave

    # Now we're comparing tile size, not full matrix size
    tile_elements = tile_rows * tile_cols
    thread_capacity = threads_per_wave * elements_per_thread

    logger.info(f"HW transpose tile: {tile_rows}x{tile_cols}={tile_elements} elements, "
                f"thread capacity: {threads_per_wave}x{elements_per_thread}={thread_capacity}")

    # Calculate thread distribution for this tile
    col_groups = ceildiv(tile_cols, elements_per_thread)
    threads_needed = tile_rows * col_groups

    if threads_needed <= threads_per_wave:
        # Standard distribution: all rows, divided columns
        threads_per_col_group = tile_rows
    else:
        # More columns than we can handle: need multiple passes
        threads_per_col_group = threads_per_wave // col_groups

    logger.info(f"Thread distribution: {col_groups} col groups, "
                f"{threads_per_col_group} threads per group")

    # Extract global offsets (workgroup and block offsets)
    global_index = remove_thread_indexing(original_index)

    # Create new thread distribution
    linear_id = hardware_constraint.linearized_thread_id

    # Thread to tile position mapping
    thread_row = linear_id % threads_per_col_group
    col_group = linear_id // threads_per_col_group
    thread_col = col_group * elements_per_thread

    # Build new index
    new_thread_index = {}
    last_two_dims = list(effective_shape[-2:])

    new_thread_index[last_two_dims[0]] = IndexSequence(thread_row, 1, 1)
    new_thread_index[last_two_dims[1]] = IndexSequence(thread_col, elements_per_thread, 1)

    # Combine global + thread parts
    final_index = combine_index(
        global_index,
        new_thread_index,
        last_two_dims[1],  # fastest_dim (column)
        elements_per_thread
    )

    # Handle any remaining dimensions
    for dim, index_seq in original_index.items():
        if dim not in final_index:
            final_index[dim] = index_seq

    return final_index

# def create_generic_hardware_transpose_index(
#     original_index: dict, read: Read, constraints: list[Constraint]
# ) -> dict:
#     hardware_constraint = get_hardware_constraint(constraints)
#     constraint_tile_size = {
#         c.dim: c.tile_size for c in constraints
#         if isinstance(c, (TilingConstraint, WorkgroupConstraint))
#     }

#     effective_shape = transpose_last2(read.type.symbolic_shape)
#     materialized_shape = materialize_shape(constraint_tile_size, effective_shape)

#     num_rows, num_cols = materialized_shape[-2:]
#     elements_per_thread = read.elements_per_thread
#     threads_per_wave = hardware_constraint.threads_per_wave

#     total_elements = num_rows * num_cols
#     max_elements_per_load = threads_per_wave * elements_per_thread

#     if subs_idxc(total_elements) <= subs_idxc(max_elements_per_load):
#         col_groups = ceildiv(num_cols, elements_per_thread)
#         threads_per_col_group = threads_per_wave // col_groups
#     else:
#         threads_per_col_group = min(num_rows, threads_per_wave)
#         col_groups = ceildiv(threads_per_wave, threads_per_col_group)

#     logger.info(f"Generic HW transpose: {num_rows}x{num_cols} (total={total_elements}), "
#                 f"threads={threads_per_wave}, col_groups={col_groups}, "
#                 f"threads_per_col_group={threads_per_col_group}")

#     global_index = remove_thread_indexing(original_index)

#     linear_id = hardware_constraint.linearized_thread_id

#     thread_row = linear_id % threads_per_col_group
#     col_group = linear_id // threads_per_col_group
#     thread_col = (col_group % col_groups) * elements_per_thread

#     thread_row = thread_row % num_rows  # Wrap if more threads than rows
#     thread_col = min(thread_col, num_cols - elements_per_thread)  # Clamp columns

#     new_thread_index = {}
#     last_two_dims = list(effective_shape[-2:])  # Get actual dimension symbols

#     new_thread_index[last_two_dims[0]] = IndexSequence(thread_row, 1, 1)      # Row
#     new_thread_index[last_two_dims[1]] = IndexSequence(thread_col, elements_per_thread, 1)  # Col

#     final_index = combine_index(
#         global_index,
#         new_thread_index,
#         last_two_dims[1],  # fastest_dim (column)
#         elements_per_thread
#     )

#     for dim, index_seq in original_index.items():
#         if dim not in final_index:
#             final_index[dim] = index_seq

#     return final_index

# def create_generic_hardware_transpose_index(
#     original_index: dict, read: Read, constraints: list[Constraint]
# ) -> dict:
#     hardware_constraint = get_hardware_constraint(constraints)
#     constraint_tile_size = {
#         c.dim: c.tile_size for c in constraints
#         if isinstance(c, (TilingConstraint, WorkgroupConstraint))
#     }

#     effective_shape = transpose_last2(read.type.symbolic_shape)
#     materialized_shape = materialize_shape(constraint_tile_size, effective_shape)

#     num_rows, num_cols = materialized_shape[-2:]
#     elements_per_thread = read.elements_per_thread
#     threads_per_wave = hardware_constraint.threads_per_wave

#     total_elements = num_rows * num_cols
#     max_elements_per_load = threads_per_wave * elements_per_thread

#     if subs_idxc(total_elements) <= safe_subs(max_elements_per_load):
#         col_groups = ceildiv(num_cols, elements_per_thread)
#         threads_per_col_group = threads_per_wave // col_groups
#     else:
#         threads_per_col_group = min(num_rows, threads_per_wave)
#         col_groups = ceildiv(threads_per_wave, threads_per_col_group)

#     logger.info(f"Generic HW transpose: {num_rows}x{num_cols} (total={total_elements}), "
#                 f"threads={threads_per_wave}, col_groups={col_groups}, "
#                 f"threads_per_col_group={threads_per_col_group}")

#     global_index = remove_thread_indexing(original_index)

#     linear_id = hardware_constraint.linearized_thread_id

#     thread_row = linear_id % threads_per_col_group
#     col_group = linear_id // threads_per_col_group
#     thread_col = (col_group % col_groups) * elements_per_thread

#     thread_row = thread_row % num_rows  # Wrap if more threads than rows
#     thread_col = min(thread_col, num_cols - elements_per_thread)  # Clamp columns

#     new_thread_index = {}
#     last_two_dims = list(effective_shape[-2:])  # Get actual dimension symbols

#     new_thread_index[last_two_dims[0]] = IndexSequence(thread_row, 1, 1)      # Row
#     new_thread_index[last_two_dims[1]] = IndexSequence(thread_col, elements_per_thread, 1)  # Col

#     final_index = combine_index(
#         global_index,
#         new_thread_index,
#         last_two_dims[1],  # fastest_dim (column)
#         elements_per_thread
#     )

#     for dim, index_seq in original_index.items():
#         if dim not in final_index:
#             final_index[dim] = index_seq

#     return final_index

# trying to learn from in thread transpose
# def create_generic_hardware_transpose_index(
#     original_index: dict, read: Read, constraints: list[Constraint]
# ) -> dict:
#     """
#     Generic hardware transpose index using in_thread_transpose.py patterns
#     """
#     # Extract problem parameters
#     hardware_constraint = get_hardware_constraint(constraints)
#     constraint_tile_size = {
#         c.dim: c.tile_size for c in constraints
#         if isinstance(c, (TilingConstraint, WorkgroupConstraint))
#     }

#     effective_shape = transpose_last2(read.type.symbolic_shape)
#     materialized_shape = materialize_shape(constraint_tile_size, effective_shape)

#     num_rows, num_cols = materialized_shape[-2:]  # [32, 16]
#     elements_per_thread = read.elements_per_thread  # 8
#     threads_per_wave = hardware_constraint.threads_per_wave  # 64

#     total_elements = num_rows * num_cols  # 32 * 16 = 512
#     max_elements_per_load = threads_per_wave * elements_per_thread  # 64 * 8 = 512

#     if total_elements > max_elements_per_load:
#         raise ValueError(f"Matrix too large: {total_elements} > {max_elements_per_load}")

#     col_groups = ceildiv(num_cols, elements_per_thread)  # 16/8 = 2
#     threads_per_col_group = threads_per_wave // col_groups  # 64/2 = 32

#     logger.info(f"Generic HW transpose: {num_rows}x{num_cols}, "
#                 f"threads={threads_per_wave}, col_groups={col_groups}, "
#                 f"threads_per_col_group={threads_per_col_group}")

#     global_index = remove_thread_indexing(original_index)

#     linear_id = hardware_constraint.linearized_thread_id

#     tiled_shape = [
#         ceildiv(num_rows, threads_per_col_group),  # [32/32] = [1]
#         ceildiv(num_cols, elements_per_thread)     # [16/8] = [2]
#     ]

#     thread_row = linear_id % threads_per_col_group  # 0-31
#     col_group = linear_id // threads_per_col_group  # 0 or 1
#     thread_col = col_group * elements_per_thread    # 0 or 8

#     new_thread_index = {}
#     last_two_dims = list(effective_shape[-2:])  # Get actual dimension symbols

#     new_thread_index[last_two_dims[0]] = IndexSequence(thread_row, 1, 1)      # Row
#     new_thread_index[last_two_dims[1]] = IndexSequence(thread_col, elements_per_thread, 1)  # Col

#     final_index = combine_index(
#         global_index,
#         new_thread_index,
#         last_two_dims[1],  # fastest_dim (column)
#         elements_per_thread
#     )

#     # Handle any remaining dimensions not in last_two_dims
#     for dim, index_seq in original_index.items():
#         if dim not in final_index:
#             final_index[dim] = index_seq

#     return final_index


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

    if not feeds_mma_instruction(write):
        return False

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }

    materialized_shape = materialize_shape(
        constraint_tile_size, read.type.symbolic_shape
    )

    if any(s > 1 for s in materialized_shape[:-2]) or any(s <= 1 for s in materialized_shape[-2:]):
        logger.info(
            f"only last 2 dims transpose is supported, got {materialized_shape}"
        )
        return False

    # breakpoint()
    # -2 is 16, -1 is 32
    # 32 is K dim, 16 is non-k
    if materialized_shape[-2] % 16 != 0 or materialized_shape[-1] % 8 != 0:
        return False

    # hw transpose works on groups of 16 threads
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
    candidates = trace.walk(is_transpose_read)

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
                    mark_hw_transpose(write, new_writes, read, new_reads, constraints)

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

def modify_index_for_full_coverage(original_index: dict, constraints: list[Constraint]) -> dict:
    """
    Modify the index to access all 32 rows instead of just 16
    Change Mod($T0, 16) to Mod($T0, 32) in the N dimension
    """

    modified_index = {}
    for dim, index_seq in original_index.items():
        if dim.name == 'N': 
            start_expr = index_seq.start

            modified_expr = start_expr.subs(
                sympy.Mod(THREAD_0, 16),
                sympy.Mod(THREAD_0, 32)
            )

            modified_index[dim] = IndexSequence(
                modified_expr,
                index_seq.size,
                index_seq.stride
            )
        elif dim.name == 'K': #fix out of bounds
              start_expr = index_seq.start

              old_pattern = 8 * sympy.floor(sympy.Mod(THREAD_0, 64) / 16)
              new_pattern = 8 * sympy.floor(THREAD_0 / 32) 

              modified_expr = start_expr.subs(old_pattern, new_pattern)
              modified_index[dim] = IndexSequence(modified_expr, index_seq.size, index_seq.stride)
        else:
            modified_index[dim] = index_seq

    return modified_index

def mark_hw_transpose(write: Write, new_writes: dict, read: Read, new_reads, constraints):
    """Mark memory for hardware transpose and create new read with transpose flag"""
    # Mark the memory for hardware transpose
    dest = get_custom(write.memory)
    dest.update_arg("hardware_transpose", LDSTransposeRead.tr8_b64)
    
    # Create new write for hardware transpose  
    with write.graph.inserting_before(write.fx_node):
        hw_write = Write(
            write.register_,
            write.memory,
            write.elements_per_thread,
            mapping=write.mapping,
            mapping_dynamic_vals=write.mapping_dynamic_vals,
        ).add_to_graph(write.graph)
        hw_write.index = write.index
        new_writes[write.memory].append(hw_write)
    
    # Create new read with transpose flag to prevent gather/scatter construction
    with read.graph.inserting_before(read.fx_node):
        new_read = Read(
            read.memory,
            read.elements_per_thread,
            mapping=read.mapping,
            mapping_dynamic_vals=read.mapping_dynamic_vals,
        ).add_to_graph(read.graph)
        # Critical: set transpose = True to prevent gather/scatter in handle_read
        new_read.transpose = True
        new_read.index = read.index
        new_read_custom = get_custom(new_read)
        new_read_custom.infer_type()
        if read.mapping_dynamic_vals:
            update_read_mapping_dynamic_values(new_read_custom)
        # Initialize as list if not exists, then append
        if read.fx_node not in new_reads:
            new_reads[read.fx_node] = []
        new_reads[read.fx_node].append(new_read)
    
    logger.info(f"Marked hardware transpose for memory: {dest}")