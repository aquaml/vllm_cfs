"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
from vllm._C import cache_ops
import time
import os
import aqua

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
AQUACache = Tuple[aqua.responsive_tensor, aqua.responsive_tensor]

use_gpu_column = os.getenv("GPU_COL") == "1"

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """
    # Initialize scratchspace for columnar swapping
    scratch_space = 0 # number of blocks
    
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Skip initializing CUDA stream and buffer for Neuron backend.
        if is_neuron():
            return

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        logger.info("Setting up aqua policy")
        responsive_policy = aqua.dynamic_policy.dynamic_policy("localhost", 8080)
        aqua.responsive_manager.set_policy(responsive_policy)

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        if use_gpu_column:
            self.cpu_cache = self.allocate_remote_gpu_cache("aqua")
            logger.info("USING GPU COLUMN")
        else:
            self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

        logger.info("Num layers: {}".format(self.num_layers))
        


    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda:0",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda:0",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    def allocate_remote_gpu_cache(self, gpu_device: str) -> List[AQUACache]:
        remote_gcache: List[AQUACache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        gpu_device = 'cpu'

        key_block_tensors = []
        value_block_tensors = []

        for _ in range(self.num_cpu_blocks):
            key_blocks = torch.empty(
                size=(self.num_layers, *key_block_shape),
                dtype=self.dtype,
                device=gpu_device,
            )
            value_blocks = torch.empty(
                size=(self.num_layers, *value_block_shape),
                dtype=self.dtype,
                device=gpu_device,
            )
            key_block_tensors.append(key_blocks)
            value_block_tensors.append(value_blocks)
        
        key_block_rts = aqua.responsive_manager.to_responsive_tensors(key_block_tensors)
        value_block_rts = aqua.responsive_manager.to_responsive_tensors(value_block_tensors)

        for key_blocks, value_blocks in zip(key_block_rts, value_block_rts):
            remote_gcache.append((key_blocks, value_blocks))
        return remote_gcache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
                device="cpu",
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
                device="cpu",
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        from vllm._C import cache_ops

        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)
    
    def _del_temp_columns(self, used_columns, temp_columns):
        if used_columns:
            torch.cuda.synchronize()
            for c in used_columns:
                temp_columns.append(c)
        
        for temp_column_key_block, temp_column_value_block in temp_columns:
            del temp_column_key_block
            del temp_column_value_block

    def _get_tensor_size_in_bytes(self, tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.numel()

    def _remote_swap_out(self, src: List[KVCache], dst: List[AQUACache], src_to_dst: Dict[int, int]) -> None:
        # We can allocate CacheEngine.scratch_space number of free blocks
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
           
        temp_column_key_block = torch.empty(
            size=(self.num_layers, *key_block_shape),
            dtype=self.dtype,
            device="cuda",
        )
        temp_column_value_block = torch.empty(
            size=(self.num_layers, *value_block_shape),
            dtype=self.dtype,
            device="cuda",
        )

        local_copy_time = 0
        for src_block_idx in src_to_dst:
            # copy every layers data into the tensor
            st = self._get_time_in_millis()
            for layer in range(self.num_layers):
                layer_temp_key_tensor = temp_column_key_block[layer]
                layer_temp_value_tensor = temp_column_value_block[layer]

                src_key_cache, src_value_cache = src[layer]
                src_layer_key_block = src_key_cache[src_block_idx]
                src_layer_value_block = src_value_cache[src_block_idx]

                layer_temp_key_tensor.copy_(src_layer_key_block)
                layer_temp_value_tensor.copy_(src_layer_value_block)
            # key_caches = [key_cache for key_cache, _ in src]
            # value_caches = [value_cache for _, value_cache in src]
            # cache_ops.gather_block_to_layers(temp_column_key_block, key_caches, temp_column_value_block, value_caches, self._get_tensor_size_in_bytes(temp_column_key_block[0]), src_block_idx, self.num_layers)
            # torch.cuda.synchronize()
            et = self._get_time_in_millis()
            local_copy_time += (et - st)
            # copy the temp tensor into the destination
            dst_key_cache, dst_value_cache = dst[src_to_dst[src_block_idx]] 
            dst_key_cache = dst_key_cache.to_torch_tensor()
            dst_value_cache = dst_value_cache.to_torch_tensor()
            # cache_ops.swap_tensors(temp_column_key_block, dst_key_cache, self._get_tensor_size_in_bytes(temp_column_key_block))
            dst_key_cache.copy_(temp_column_key_block)
            # cache_ops.swap_tensors(temp_column_value_block, dst_value_cache, self._get_tensor_size_in_bytes(temp_column_value_block))
            dst_value_cache.copy_(temp_column_value_block)   

        # print("LOCAL SWAP OUT: {}".format(local_copy_time))

    def _get_time_in_millis(self):
        return int(time.time() * 1000)

    def _remote_swap_in(self, src: List[AQUACache], dst: List[KVCache], src_to_dst: Dict[int, int]) -> None:
        # We can allocate CacheEngine.scratch_space number of free blocks
        # src is aqua tensor
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
   
        temp_column_key_block = torch.empty(
            size=(self.num_layers, *key_block_shape),
            dtype=self.dtype,
            device="cuda",
        )
        temp_column_value_block = torch.empty(
            size=(self.num_layers, *value_block_shape),
            dtype=self.dtype,
            device="cuda",
        )

        local_copy_time = 0
        remote_copy_time = 0

        for src_block_idx in src_to_dst:
            # column tensors in remote swap space
            remote_column_key_block, remote_column_value_block = src[src_block_idx]
            temp_column_key_block.copy_(remote_column_key_block.to_torch_tensor())
            temp_column_value_block.copy_(remote_column_value_block.to_torch_tensor())

            st = self._get_time_in_millis()
            key_caches = [key_cache for key_cache, _ in dst]
            value_caches = [value_cache for _, value_cache in dst]
            cache_ops.scatter_block_to_layers(temp_column_key_block, key_caches, temp_column_value_block, value_caches, self._get_tensor_size_in_bytes(temp_column_key_block[0]), src_to_dst[src_block_idx], self.num_layers)
            torch.cuda.synchronize()
            et = self._get_time_in_millis()
            local_copy_time += (et - st)

        # logger.info("LOCAL COPY TIMES: {}, REMOTE COPY TIMES: {}".format(local_copy_time, remote_copy_time))

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        if use_gpu_column:
            with torch.cuda.stream(self.cache_stream):
                self._remote_swap_in(self.cpu_cache, self.gpu_cache, src_to_dst)
        else:
            self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)


    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        if use_gpu_column:
            with torch.cuda.stream(self.cache_stream):
                self._remote_swap_out(self.gpu_cache, self.cpu_cache, src_to_dst)
        else:
            self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)
            
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        from vllm._C import cache_ops

        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
