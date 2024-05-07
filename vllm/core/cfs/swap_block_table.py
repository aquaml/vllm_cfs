from vllm.core.block_manager import BlockSpaceManager
from vllm.sequence import (Sequence, SequenceGroup)
from vllm.prefix import Prefix
from typing import Dict, List, Set
from vllm.block import PhysicalTokenBlock

# List to store swap btes (Block Table Entries)
SwapBlockTable = List[PhysicalTokenBlock]

class SwapBlockMM:
    def __init__(self, block_manager: BlockSpaceManager) -> None:
        self.block_manager = block_manager
        self.seq_sbts: Dict[SequenceGroup, SwapBlockTable] = {}
        self.prefix_sbts: Dict[Prefix, SwapBlockTable] = {}
        self.swapped_groups: Set[SequenceGroup] = set()
    
    def _swap_out_prefix(self, prefix: Prefix, blocks_to_swap_out: Dict[int, int]) -> None:
        # This happens only once for a sequence group, across multiple swap ins and swap outs
        if prefix in self.prefix_sbts:
            return
        
        self.prefix_sbts[prefix] = []
        prefix_sbt = self.prefix_sbts[prefix]
        
        # hbm blocks of prefix
        hbm_blocks = prefix.block_table

        for virtual_block_idx in range(len(hbm_blocks)):
            swap_block: PhysicalTokenBlock = self.block_manager.cpu_allocator.allocate()
            swap_block.ref_count -= 1 # Individual sequences referencing this prefix will increment this
            hbm_block: PhysicalTokenBlock = hbm_blocks[virtual_block_idx] # No need to free the hbm block here, individual sequences will do it
            prefix_sbt.append(swap_block)
            blocks_to_swap_out[hbm_block.block_number] = swap_block.block_number
    
    def _swap_out_seq(self, seq: Sequence, blocks_to_swap_out: Dict[int, int], prefix: Prefix, swap_block_cache: Dict[int, int]) -> None:
        if seq not in self.seq_sbts:
            self.seq_sbts[seq] = []

        seq_sbt = self.seq_sbts[seq]
        # Block table mapping virtual blocks to local HBM memory
        hbm_block_table = self.block_manager.block_tables[seq.seq_id]

        prefix_blocks_count = 0
        if prefix is not None and prefix.allocated:
            assert prefix.length < len(hbm_block_table)
            prefix_sbt = self.prefix_sbts[prefix]
            prefix_block_table = prefix.block_table
            
            # Every sequence is referencing it
            # Update block table entries
            for swap_block in prefix_sbt:
                swap_block.ref_count += 1

            # Swap only the blocks exclusive to this sequence
            prefix_blocks_count = len(prefix_block_table)

        # always swap out the last block because it is overwritten, aka dirty
        if len(seq_sbt) > 0:
            blocks_to_swap_out[hbm_block_table[prefix_blocks_count + len(seq_sbt) - 1].block_number] = seq_sbt[-1].block_number

        new_sbt = []
        for virutal_block_idx in range(0, len(hbm_block_table)):
            hbm_block: PhysicalTokenBlock = hbm_block_table[virutal_block_idx]
            swap_block: PhysicalTokenBlock = None
            reusing_block = False
            # Doing this to handle beam search
            # if the same hbm block has been referenced before, then use the swap block associated with it
            if hbm_block.block_number in swap_block_cache:
                swap_block = swap_block_cache[hbm_block.block_number]
                swap_block.ref_count += 1

            # Else, allocate a new block
            else:
                if virutal_block_idx >= len(seq_sbt):
                    swap_block = self.block_manager.cpu_allocator.allocate()
                else:
                    swap_block = seq_sbt[virutal_block_idx]
                    reusing_block = True
                swap_block_cache[hbm_block.block_number] = swap_block
            
            new_sbt.append(swap_block)
            if not reusing_block:
                blocks_to_swap_out[hbm_block.block_number] = swap_block.block_number
        self.seq_sbts[seq] = new_sbt

        # Free every hbm
        for hbm_block in hbm_block_table:
            self.block_manager.gpu_allocator.free(hbm_block)

    def swap_out(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int]) -> None:
        assert seq_group.num_seqs() > 0
        # print("swapping out: {}".format(seq_group.request_id))
        if seq_group not in self.swapped_groups:
            self.swapped_groups.add(seq_group)

        if seq_group.prefix is not None and seq_group.prefix.allocated:
            # print("Swapping out prefix for: {}".format(seq_group.request_id))
            self._swap_out_prefix(seq_group.prefix, blocks_to_swap_out)

        # TODO: Fix swap block cache for beam search
        swap_block_cache: Dict[int, int] = {}
        for seq in seq_group.get_seqs():
            self._swap_out_seq(seq, blocks_to_swap_out, seq_group.prefix, swap_block_cache)

        del swap_block_cache
        pass

    def _swap_in_prefix(self, prefix: Prefix, blocks_to_swap_in: Dict[int, int]) -> None:
        assert prefix in self.prefix_sbts
        prefix_sbt = self.prefix_sbts[prefix]
        assert len(prefix_sbt) == len(prefix.block_table)
        
        for virtual_block_idx, swap_block in enumerate(prefix_sbt):
            hbm_block = self.block_manager.gpu_allocator.allocate()
            hbm_block.ref_count -= 1
            prefix.block_table[virtual_block_idx] = hbm_block
            blocks_to_swap_in[swap_block.block_number] = hbm_block.block_number


    def _swap_in_sequence(self, seq: Sequence, blocks_to_swap_in: Dict[int, int], prefix: Prefix, hbm_block_cache: Dict[int, int]) -> None:
        seq_sbt = self.seq_sbts[seq]
        number_of_sbts = len(seq_sbt)

        # Block table mapping virtual blocks to local HBM memory
        hbm_block_table = self.block_manager.block_tables[seq.seq_id]

        prefix_blocks = 0
        # handle prefix swapping
        if prefix is not None and prefix.allocated:
            for virtual_block_idx, hbm_block in enumerate(prefix.block_table):
                hbm_block.ref_count += 1
                hbm_block_table[virtual_block_idx] = hbm_block

            prefix_blocks = len(prefix.block_table)
        
        # print("prefix blocks: {}, num_sbts: {}, hbm_blocks: {}".format(prefix_blocks, number_of_sbts, len(hbm_block_table)))
        assert prefix_blocks + number_of_sbts == len(hbm_block_table)
        
        for virtual_block_idx in range(prefix_blocks, len(seq_sbt)):
            hbm_block: PhysicalTokenBlock = None
            swap_block: PhysicalTokenBlock = seq_sbt[virtual_block_idx - prefix_blocks]
            if swap_block.block_number not in hbm_block_cache:
                hbm_block = self.block_manager.gpu_allocator.allocate()
                hbm_block_cache[swap_block.block_number] = hbm_block
            else:
                hbm_block = hbm_block_cache[swap_block.block_number]
                hbm_block.ref_count += 1

            hbm_block_table[virtual_block_idx] = hbm_block
            blocks_to_swap_in[swap_block.block_number] = hbm_block.block_number

        
    def swap_in(self, seq_group: SequenceGroup, blocks_to_swap_in: Dict[int, int]) -> None:
        assert seq_group.num_seqs() > 0
        # print("swapping in: {}".format(seq_group.request_id))
        # TODO: Handle sequence groups with waiting and running sequences
        if seq_group.prefix is not None and seq_group.prefix.allocated:
            print("swapping in prefix")
            self._swap_in_prefix(seq_group.prefix, blocks_to_swap_in)

        hbm_block_cache: Dict[int, int] = {}
        for seq in seq_group.get_seqs():
            self._swap_in_sequence(seq, blocks_to_swap_in, seq_group.prefix, hbm_block_cache)
        
        del hbm_block_cache
        pass

    def is_swapped_before(self, seq_group: SequenceGroup):
        return seq_group in self.swapped_groups

    def free(self, seq_group: SequenceGroup) -> None:
        for seq in seq_group.get_seqs():
            if seq in self.seq_sbts:
                seq_sbt = self.seq_sbts[seq]
                for block in seq_sbt:
                    self.block_manager.cpu_allocator.free(block)
                del self.seq_sbts[seq]
        if seq_group in self.swapped_groups:
            self.swapped_groups.remove(seq_group)
        # TODO: Handle prefix sbt
        
