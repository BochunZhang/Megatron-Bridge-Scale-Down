# 安装说明

## 1. 基于 uv 安装 Megatron-Bridge & Megatron-LM

Megatron-Bridge & Megatron-LM 使用 uv 管理项目的 python 环境, 其环境配置最好放在 docker 虚拟机里面直接完成.

流程为:
1. 拉取 docker, 已配置 python=3.12, torch==2.13.0, torchvision==0.28.0, triton==3.7.1, 这些 package 安装在 system python 的 site-packages 里面, uv venv 时配置了 --system-site-packages, 因此不会被 uv sync 删除
2. 修改 torch 源码, 增加一个 patch
3. 在 sytem python 里面安装 hybrid-ep, 拉取源码后也需要执行一个 patch 后再编译
4. 安装 Megatron-Bridge, 其中 transformer-engine 锁定为 2.17 版本, 且需要从源码编译

Megatron-LM 的 main 分支使用 transformer engine 2.18 + nccl 2.30.7+, 支持新特性 nccl_ep, core_v0.19.0 使用 transformer engine 2.17.

Megatron-LM 虚拟机默认使用 hybrid-ep, 而非 deep-ep, 为测试不同分支, 我们统一配置为
- python 3.12.12 (通过 uv 下载 clang 编译版本), 存储在 .python 路径下
- torch==2.13.0, torchvision==0.28.0, triton==3.7.1
- transformer_engin 2.18 + nccl 2.30.7
- transformers 最新版本
- .tmp 下
- .env-base 存放 python3.12-torch2.13, 安装 torch==2.13.0, torchvision==0.28.0, triton==3.7.1
- .venv/python3.12-torch2.13-engin2.18-hybrid-ep 存放 hybrid ep 虚拟环境
- .venv/python3.12-torch2.13-engin2.18-deep-ep 存放 deep ep 虚拟环境

关于 torch & transformer-engine 的编译安装
1. 官方提供的 torch 自带 nccl, 封装在 nvidia-nccl-cu13 里面, 可以通过 ldd 检查 torch 的 lib 库, 里面的 nccl 链接到了这个 package 里面
2. transformer-engine 应该使用和 torch 相同版本的 nccl 编译安装, 最好通过 NCCL_HOME 来指向 torch 使用的 nccl 源码, 避免找不到 nccl_device.h 
3. transformer-engine 会默认编译 NCCL-EP 扩展, 但是该功能需要新版本的 nccl, e.g. nccl 2.30.7+, 如果 torch 默认版本较低, 编译该扩展会报错, 因此要使用 NVTE_WITH_NCCL_EP 关闭 NCCL-EP 扩展编译




注意:
1. uv 下载的 python 无法安装 package, 也就无法通过 virtualenv 创建虚拟环境, 但可以使用 python -m venv 来创建虚拟环境
2. uv 创建的虚拟环境不会在激活时替换 pip, 这意味着 pip install 操作的是系统 python, --system-site-packages 表示使用系统的 python, 因此可以在 python 里面安装 package
3. uv 环境激活时, 会先退出已经激活的 activate, pip 显示的是系统 python 安装的内容, uv pip 显示的是 uv 环境里面安装的 package
  - python / uv run python 都无法找到系统安装的环境, 不知道是否因为不是使用系统 python 创建的虚拟环境
4. 解决方案是使用
  - virutalenv/venv 构建虚拟环境, source activate 会自动激活
  - 不要用 uv 创建虚拟环境, 否则 pip install 和 uv pip install 操作的是不同的 site-packages


应该按照如下流程来实现
```bash
# Set environment variables for GB200
export NVTE_BUILD_NUM_PHILOX_ROUNDS=3
export TORCH_CUDA_ARCH_LIST="10.0"  # Blackwell architecture
export NVCC_THREADS=16
export FLASH_MLA_DISABLE_SM90=1

# Set UV environment
export UV_HTTP_TIMEOUT=120
export UV_LINK_MODE=copy

export CFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"
export CXXFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE

# 1. 安装 python
uv python install 3.12.12 --install-dir .python/ --trusted-host https://github.com

# 2. 创建虚拟环境
mkdir -p .env-base
.python/cpython-3.12.12-linux-aarch64-gnu/bin/python -m venv .python/python-3.12.12
source .python/python3.12.12/bin/activate
pip install virtualenv
deactivate

# 3.1 在激活的环境里面创建 hybrid-ep venv
source .python/python3.12.12/bin/activate
virtualenv -p .python/python3.12.12/bin/python .venv/python3.12-torch2.13-engin2.18-hybrid-ep
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> ".venv/python3.12-torch2.13-engin2.18-hybrid-ep/bin/activate"
source .venv/python3.12-torch2.13-engin2.18-hybrid-ep/bin/activate
echo $VIRTUAL_ENV
echo $UV_PROJECT_ENVIRONMENT

source .python/python3.12.12/bin/activate
virtualenv -p .python/python3.12.12/bin/python .venv/python3.12-torch2.13-engin2.17-hybrid-ep
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> ".venv/python3.12-torch2.13-engin2.17-hybrid-ep/bin/activate"
source .venv/python3.12-torch2.13-engin2.17-hybrid-ep/bin/activate
echo $VIRTUAL_ENV
echo $UV_PROJECT_ENVIRONMENT

# 3.2 在激活的环境里面创建 deep-ep venv
source .python/python3.12.12/bin/activate
virtualenv -p .python/python3.12.12/bin/python .venv/python3.12-torch2.13-engin2.18-deep-ep
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> ".venv/python3.12-torch2.13-engin2.18-deep-ep/bin/activate"
source .venv/python3.12-torch2.13-engin2.18-deep-ep/bin/activate
echo $VIRTUAL_ENV
echo $UV_PROJECT_ENVIRONMENT

# 4. 安装 torch 并安装 torch patch
pip install triton==3.7.1 torch==2.13.0 torchvision==0.28.0
pip install nvidia-cutlass-dsl[cu13]==4.5.0

# 应用 torch patch
bash install/torch-patch.sh

# 测试
python - <<'PY'
import torch
from torch.library import Library, _clear_torch_ops_cache

namespace_name = "_mbridge_finalizer_test"
library = Library(namespace_name, "DEF")
library.define("uncached_op() -> None")
namespace = getattr(torch.ops, namespace_name)
original_getattr = type(namespace).__getattr__


def fail_getattr(self, name):
    raise AssertionError(f"Torch op cache cleanup called __getattr__ for {name}")


type(namespace).__getattr__ = fail_getattr
try:
    _clear_torch_ops_cache({f"{namespace_name}::uncached_op"})
finally:
    type(namespace).__getattr__ = original_getattr
PY

NCCL_DIR=$(python3 -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null)
export NCCL_HOME="$NCCL_DIR"
export CPLUS_INCLUDE_PATH="$NCCL_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LD_LIBRARY_PATH="$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

uv sync --only-group build --inexact
NVTE_WITH_NCCL_EP=0 uv sync --link-mode copy --all-extras --all-groups --no-group diffusion --inexact
uv sync --upgrade-package transformers --inexact


bash install/install-hybrid-ep.sh
bash install/install-deep-ep.sh
```


## 2. NCCL 版本选择

如何切换 nccl 版本?
1. 安装 torch 的时候会安装 nvidia-nccl-cu13 这个 python package, torch 会静态链接到 nccl 
2. 切换 nccl 版本需要从头编译 torch


## 3. uv 升级 package 版本

如何升级 package?
1. echo "transformers==5.16.1" | uv pip compile - 查询 package 的依赖关系
2. uv add "transformers>=5.8,<=5.16.1" "tokenizers>=0.22.0,<=0.23.2" --no-syn 解析升级后的需求

todo: DeepEPv2 安装失败, 推荐使用 nccl 2.30.7+ ...


```python
output_layer_cls = (
    TELMHeadColumnParallelLinear
    if is_mxfp8_output_proj_active(config)
    else tensor_parallel.ColumnParallelLinear
)
```

## 4. Apex 编译安装

安装 Apex

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 25.09

# Build with core extensions (cpp and cuda)
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
# To build with additional extensions, specify them with environment variables
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation .
# To build all contrib extensions at once
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_ALL_CONTRIB_EXT=1 pip install -v --no-build-isolation .
```

- GPTModel 的 output_layer 会在使用 bf16 训练时使用 ColumnParallelLinear, 这是 Megatron 基于 torch.nn.Module 构建的 Linear 类, 它基于 Apex 提供 gradient_accumulation_fusion 等 fused 功能, 因此需要安装 Apex 来获得更好的性能
- Apex 的 setup 里面硬编码支持 compute_70, 但是最新的 cuda 13.0 已经移除对 compute_70 的支持, 导致 nvcc 报错, 即使明确只编译 sm=10.0, 硬编码仍会将 sm=7.0 作为编译目标
- 解决方案: 修改 Apex 的 setup.py, 只保留 compute_100 作为编译目标
- 直接使用基础 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation . 编译即可


## 5. DeepSpeed 安装

DeepSpeed 可以通过 pip 直接安装, 但是 DeepSpeed-Examples 提供的测试需要安装 flash-attn 等 package

```bash
export NVTE_BUILD_NUM_PHILOX_ROUNDS=3
export TORCH_CUDA_ARCH_LIST="10.0"  # Blackwell architecture
export NVCC_THREADS=16
export FLASH_MLA_DISABLE_SM90=1

# Set UV environment
export UV_HTTP_TIMEOUT=120
export UV_LINK_MODE=copy

export CFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"
export CXXFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE

# 3.2 在激活的环境里面创建 deep-ep venv
source .python/python3.12.12/bin/activate
virtualenv -p .python/python3.12.12/bin/python .venv/python3.12-torch2.13-engin2.18-deep-ep
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> ".venv/python3.12-torch2.13-engin2.18-deep-ep/bin/activate"
source .venv/python3.12-torch2.13-engin2.18-deep-ep/bin/activate
echo $VIRTUAL_ENV
echo $UV_PROJECT_ENVIRONMENT

# 4. 安装 torch 并安装 torch patch
pip install triton==3.7.1 torch==2.13.0 torchvision==0.28.0
pip install nvidia-cutlass-dsl[cu13]==4.5.0

# 应用 torch patch
bash install/torch-patch.sh

# 测试
python - <<'PY'
import torch
from torch.library import Library, _clear_torch_ops_cache

namespace_name = "_mbridge_finalizer_test"
library = Library(namespace_name, "DEF")
library.define("uncached_op() -> None")
namespace = getattr(torch.ops, namespace_name)
original_getattr = type(namespace).__getattr__


def fail_getattr(self, name):
    raise AssertionError(f"Torch op cache cleanup called __getattr__ for {name}")


type(namespace).__getattr__ = fail_getattr
try:
    _clear_torch_ops_cache({f"{namespace_name}::uncached_op"})
finally:
    type(namespace).__getattr__ = original_getattr
PY

NCCL_DIR=$(python3 -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null)
export NCCL_HOME="$NCCL_DIR"
export CPLUS_INCLUDE_PATH="$NCCL_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LD_LIBRARY_PATH="$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

uv sync --only-group build --inexact
NVTE_WITH_NCCL_EP=0 uv sync --link-mode copy --all-extras --all-groups --no-group diffusion --inexact
uv sync --upgrade-package transformers --inexact

MAX_JOBS=4 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS="100" pip install flash-attn --no-build-isolation

pip install deepspeed
```

注意:
- 限制 MAX_JOBS 和 NVCC_THREADS, 否则并行编译会占满全部 CPU, 导致设备卡死和 OOM, 但是报错与 OOM 无关, 很难 debug
- 限制架构, 减少编译耗时