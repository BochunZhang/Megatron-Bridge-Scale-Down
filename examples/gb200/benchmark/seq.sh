#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Sequence-length helpers for the model launchers (source this file).
# Same grammar as mbench.run.parse_sequence_length: k = 1024, m = 1024**2.

# 4k | 16K | 256k | 1m | 4096 -> token count on stdout; returns 2 on bad input.
mbench_seq_tokens() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$value" =~ ^([1-9][0-9]*)k$ ]]; then
    echo $((BASH_REMATCH[1] * 1024))
  elif [[ "$value" =~ ^([1-9][0-9]*)m$ ]]; then
    echo $((BASH_REMATCH[1] * 1048576))
  elif [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value"
  else
    echo "invalid --seq '$1' (expected e.g. 4k, 16k, 64k, 256k or a token count)" >&2
    return 2
  fi
}

# Token count -> canonical result-directory tag: 4096 -> 4k, 262144 -> 256k, 1048576 -> 1m, 5000 -> 5000.
mbench_seq_tag() {
  local tokens=$1
  if ((tokens % 1048576 == 0)); then
    echo "$((tokens / 1048576))m"
  elif ((tokens % 1024 == 0)); then
    echo "$((tokens / 1024))k"
  else
    echo "$tokens"
  fi
}
