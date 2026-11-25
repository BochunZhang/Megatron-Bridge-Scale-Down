// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nccl.h>
#include <nccl_device.h>

#include <new>

namespace {

struct FusedAgRsState {
  ncclComm_t comm;
  int ctas;
  ncclWindow_t ag_window = nullptr;
  ncclWindow_t rs_window = nullptr;
  ncclDevComm device_comm{};
  bool device_comm_created = false;
};

// The caller drains the CUDA stream before releasing the registered windows.
ncclResult_t release(FusedAgRsState* state) {
  ncclResult_t result = ncclSuccess;
  if (state->device_comm_created) result = ncclDevCommDestroy(state->comm, &state->device_comm);
  if (state->rs_window) {
    const ncclResult_t status = ncclCommWindowDeregister(state->comm, state->rs_window);
    if (result == ncclSuccess) result = status;
  }
  if (state->ag_window) {
    const ncclResult_t status = ncclCommWindowDeregister(state->comm, state->ag_window);
    if (result == ncclSuccess) result = status;
  }
  delete state;
  return result;
}

}  // namespace

extern "C" int fused_ag_rs_create(ncclComm_t comm, void* ag_out, size_t ag_window_bytes, void* rs_in,
                                 size_t rs_window_bytes, int ctas, void** out_state) {
  if (!out_state) return ncclInvalidArgument;
  *out_state = nullptr;
  if (!comm || !ag_out || !rs_in || ag_window_bytes == 0 || rs_window_bytes == 0 || ctas < 1 || ctas > 64) {
    return ncclInvalidArgument;
  }

  auto* state = new (std::nothrow) FusedAgRsState{comm, ctas};
  if (!state) return ncclSystemError;
  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.lsaMultimem = true;
  requirements.lsaBarrierCount = ctas;

  ncclResult_t status = ncclCommWindowRegister(comm, ag_out, ag_window_bytes, &state->ag_window,
                                               NCCL_WIN_COLL_SYMMETRIC);
  if (status == ncclSuccess) {
    status = ncclCommWindowRegister(comm, rs_in, rs_window_bytes, &state->rs_window, NCCL_WIN_COLL_SYMMETRIC);
  }
  if (status == ncclSuccess && (!state->ag_window || !state->rs_window)) {
    status = ncclInvalidUsage;
  }
  if (status == ncclSuccess) {
    status = ncclDevCommCreate(comm, &requirements, &state->device_comm);
  }
  if (status != ncclSuccess) {
    release(state);
    return status;
  }
  state->device_comm_created = true;
  *out_state = state;
  return ncclSuccess;
}

extern "C" int fused_ag_rs_launch(void* handle, const void* ag_in, size_t ag_bytes, void* rs_out, size_t rs_count,
                                 cudaStream_t stream) {
  const auto* state = static_cast<const FusedAgRsState*>(handle);
  if (!state) return ncclInvalidArgument;
  return ncclDevFusedAllGatherReduceScatter(ag_in, state->ag_window, 0, ag_bytes, state->rs_window, 0, rs_out,
                                            rs_count, &state->device_comm, state->ctas, stream);
}

extern "C" int fused_ag_rs_destroy(void* handle) {
  return handle ? release(static_cast<FusedAgRsState*>(handle)) : ncclSuccess;
}
