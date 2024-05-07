#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void swap_tensors(
  torch::Tensor& src,
  torch::Tensor& dst,
  int64_t size);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping,
  const std::string& kv_cache_dtype);

// Just for unittest
void convert_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache);

void scatter_block_to_layers(
  torch::Tensor& key_src,
  std::vector<torch::Tensor>& key_caches,
  torch::Tensor& value_src,
  std::vector<torch::Tensor>& value_caches,
  int64_t size,
  int64_t block_id,
  int64_t num_layers
);

void gather_block_to_layers(
  torch::Tensor& key_dst,
  std::vector<torch::Tensor>& key_caches,
  torch::Tensor& value_dst,
  std::vector<torch::Tensor>& value_caches,
  int64_t size,
  int64_t block_id,
  int64_t num_layers
);
