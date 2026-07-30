#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

# Default to ${PWD}/.cache/huggingface when HF_HOME is unset.
export HF_HOME=${HF_HOME:-${PWD}/.cache/huggingface}

mkdir -p models
echo "Downloading models to /offline_inference_data/models..."

# workaround to disable update check
# can be removed once hf >= 1.16.2
mkdir -p "${HF_HOME}"
touch "${HF_HOME}/.check_for_update_done"

hf download BAAI/bge-m3 \
  --local-dir models/bge-m3

echo "All models downloaded."
