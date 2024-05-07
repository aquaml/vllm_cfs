from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from typing import List

class rb_tree:
    def __init__(self) -> None:
        self.nodes: List[SequenceGroup] = []
    
    def add_node(self, seq_group: SequenceGroup) -> None:
        self.nodes.append(seq_group)
    
    def get_num_generated(self, sequence_group: SequenceGroup):
        return sum(ss.get_output_len() for ss in sequence_group.get_seqs())
    
    def has_nodes(self):
        return len(self.nodes) > 0

    def get_least(self) -> SequenceGroup:
        sorted_sequence_groups = sorted(self.nodes, key=self.get_num_generated)
        self.nodes.remove(sorted_sequence_groups[0])
        log_pointer = sorted_sequence_groups[0]
        # print("Returning sequence group: {} with total generated tokens: {}".format(log_pointer.request_id, self.get_num_generated(log_pointer)))
        return sorted_sequence_groups[0]
    
    def remove_node(self, seq_group: SequenceGroup):
        if seq_group in self.nodes:
            self.nodes.remove(seq_group)
    
    def clear_tree(self) -> None:
        self.nodes.clear()
        