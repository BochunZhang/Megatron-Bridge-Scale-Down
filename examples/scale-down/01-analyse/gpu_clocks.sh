#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

gpu_clock_mode() {
    if [[ "$1" == true ]]; then
        printf '%s\n' unlocked
    else
        printf '%s\n' tdp,tdp
    fi
}

# Call once per node before launching a GPU case.
configure_gpu_clocks() {
    local clock_arg=--lock-gpu-clocks=tdp,tdp
    if [[ "$1" == true ]]; then
        clock_arg=--reset-gpu-clocks
    fi
    nvidia-smi "${clock_arg}" || {
        local status=$?
        printf 'Failed to configure GPU clocks with nvidia-smi %s\n' "${clock_arg}" >&2
        return "${status}"
    }
}
