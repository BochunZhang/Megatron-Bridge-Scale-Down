# GB200：用 NCCL 2.32.3 源码编译 torch 2.13.0 + torchvision 0.28.0

[install-torch-2.13.0-nccl-2.32.3.sh](install-torch-2.13.0-nccl-2.32.3.sh) 按官方 aarch64 cu13 wheel 的编译选项
从源码编 torch 2.13.0 和 torchvision 0.28.0，NCCL **动态链接**到
[install-nccl-2.32.3.sh](install-nccl-2.32.3.sh) 编出来的 `nccl/nccl-2.32.3/build`，只编 GB200（sm_100）。

官方 torch 2.13.0 编译时用的是 NCCL 2.29.7（`nvidia-nccl-cu13==2.29.7`），运行时也加载这份。编完后 torch 的
编译期和运行期 NCCL 都是 2.32.3，并且进程里只有这一份 `libnccl`。

## 前置条件

在 GB200 节点（Linux aarch64）上运行，需要：

- CUDA toolkit 13.0 或 13.2，路径用 `CUDA_HOME` 或 `--cuda-home` 指定，默认 `/usr/local/cuda`。官方用的是
  13.0.2 / 13.2.1，小版本不同时脚本会警告。
- `git`、`uv`、`wget`、C/C++ 编译器（官方用 gcc 13，版本不同时脚本会警告）、能访问 GitHub 和 PyPI。
- 推荐安装 `libpng-dev libjpeg-turbo8-dev libwebp-dev`。缺少时 torchvision 会编成不带对应编解码的版本，脚本会警告。
- 磁盘：torch 源码（含子模块）、build 目录、NVPL / ACL 都放在 `.tmp/` 下，需要几十 GB。

NVPL 和 ACL 是 torch 在 aarch64 上的 BLAS / oneDNN 依赖。`/usr/local/lib`、`/acl` 下没有时，脚本会用官方的
`install_nvpl.sh` / `install_acl.sh` 装到 `.tmp/`，其中 ACL 要用 scons 编译，会多花一些时间。

## 运行步骤

在 `megatron-bridge` 根目录执行：

```bash
# 1. 编 NCCL 2.32.3 到 nccl/nccl-2.32.3/build，并检查 TMA / multicast / CE 都编进去了
bash install/install-nccl-2.32.3.sh

# 2. 新建并激活一个 Python 3.12 venv（torch 会装进这个 venv）
python3.12 -m venv .venv/python3.12-torch2.13-nccl2.32.3
source .venv/python3.12-torch2.13-nccl2.32.3/bin/activate

# 3. 编 torch + torchvision，装进当前 venv，然后跑检查
bash install/install-torch-2.13.0-nccl-2.32.3.sh 2>&1 | tee .tmp/install-torch-nccl-2.32.3.log
```

第 3 步的输出：

| 输出 | 位置 |
| --- | --- |
| torch / torchvision wheel | `.tmp/wheels/torch-2.13.0-nccl-2.32.3/torch-2.13.0+cu130-cp312-cp312-linux_aarch64.whl`、`torchvision-0.28.0+cu130-...whl` |
| 编译日志 | `.tmp/torch-build.log`、`.tmp/torchvision-build.log` |
| 编译用 venv（官方的编译依赖，不污染目标 venv） | `.tmp/torch-2.13.0-build-py3.12` |
| 官方编译环境 / 运行时依赖 | `.tmp/torch-2.13.0-build.env`、`.tmp/torch-2.13.0-runtime-requirements.txt` |

成功时最后一行是 `=== torch 2.13.0+cu130 + torchvision ready in ...`，有检查失败时脚本以非 0 退出，并打印失败项数量。

### 常用参数

| 参数 | 作用 |
| --- | --- |
| `--python PY` | 目标解释器，默认当前 venv 的 `python` |
| `--nccl-home DIR` | NCCL 2.32.3 的 build 目录，默认 `nccl/nccl-2.32.3/build` |
| `--cuda-home DIR` | CUDA toolkit，默认 `CUDA_HOME` 或 `/usr/local/cuda` |
| `--jobs N` | 编译并发数 `MAX_JOBS`，默认按官方取 `min(nproc - 2, 12)` |
| `--no-install` | 只编 wheel，不安装 |
| `--check-only` | 不编译不安装，只对 `--python` 跑检查 |

## 检查项

安装后脚本会自动检查以下内容，也可以随时用 `--check-only` 重跑：

1. CMake 实际选中的 `nccl.h` / `libnccl.so` 都在 `--nccl-home` 下（读 `build/CMakeCache.txt`）。
2. `torch.cuda.nccl.version()`（编译期）和 `get_runtime_nccl_version()`（运行期）都是 2.32.3。
3. 进程里只映射了一份 `libnccl`，并且就是 `--nccl-home/lib` 下的那份；没有安装 `nvidia-nccl-*` wheel。
4. `libtorch_cuda.so` 的 `NEEDED` 里有 `libnccl.so.2`（说明是动态链接），RPATH 里有 `--nccl-home/lib`。
5. `NCCLConfig.max_p2p_peers` 能访问（只有编译期 NCCL ≥ 2.30 才有这个绑定）。
6. torchvision 0.28.0 的 CUDA 版本与 torch 一致，PNG / JPEG 编解码可用，`ops.nms` 能在 GPU 上运行，arch list 里有 `sm_100`。
7. 有 ≥ 2 张 GPU 时跑 `all_reduce` / `all_gather_into_tensor` / `reduce_scatter_tensor`，核对结果，并确认
   `NCCL_DEBUG=VERSION` 打印的是 2.32.3。

更完整的报告：`bash install/probe-torch-nccl-version.sh --python "$(which python)"`。

## 装到其他 venv

wheel 编好后，其他 venv 不需要重新编译：

```bash
python3.12 -m venv .venv/another-env && source .venv/another-env/bin/activate
W=.tmp/wheels/torch-2.13.0-nccl-2.32.3
uv pip install $W/torch-2.13.0+cu130-*.whl $W/torchvision-0.28.0+cu130-*.whl
bash install/install-torch-2.13.0-nccl-2.32.3.sh --check-only
```

## 注意事项

- **不要删除或移动 `nccl/nccl-2.32.3/build`。** 它的绝对路径写在 `libtorch_cuda.so` 的 RPATH 里。换位置需要重新运行脚本。
- **版本号和官方 wheel 相同（`2.13.0+cu130`）。** `uv pip install --reinstall torch`、`--upgrade` 或 `uv sync`
  都可能把它换回 PyPI 上的官方版本，而且从版本号上看不出区别。做完这类操作后，要用 `--check-only` 确认 NCCL 仍是 2.32.3。
- **只有 sm_100 的 SASS，不带 PTX。** 这个 torch 只能在 GB200 / B200 这类计算能力 10.0 的 GPU 上运行，在 H100 等其他架构上
  会报 `no kernel image is available`。
- **nvshmem 会被降级。** torch 依赖 `nvidia-nvshmem-cu13==3.4.5`，安装时会把 DeepEP 用的 3.6.5 降下去，脚本会提示。
  DeepEP 需要 3.6.5 时，装完 torch 后再装一次。
- **Transformer Engine 要用同一份 NCCL 重新编译：** `NCCL_HOME=nccl/nccl-2.32.3/build`。
- 编译时脚本会清掉 shell 里的 `USE_*` / `BUILD_*` / `CMAKE_*` / `TORCH_*` / `NCCL_*` / `NVCC_*` 和 `CFLAGS` 等变量，
  避免 [installation.md](../documents/installation/installation.md) 里 export 的那些变量混进编译选项。

## 和官方 wheel 的差异

torch v2.13.0 已经删除了 `.ci/manywheel/build_cuda.sh`，改为 `build_env_setup.py`、`build_wheel.py`、`repair_wheel.py`。
脚本直接 import 克隆下来的 v2.13.0 里的这几个模块，所以下面这些都和官方一致：

- 编译开关：`USE_SYSTEM_NCCL=1`、`USE_STATIC_NCCL=0`、`TORCH_NVCC_FLAGS`、`USE_CUPTI_SO=1` 等；
- 运行时固定依赖：cuda-toolkit 13.0.3、cuDNN 9.20.0.48、cuSPARSELt 0.8.1、NVSHMEM 3.4.5、triton 3.7.1；
- RPATH 结构，以及 aarch64 下 NVPL / ACL / libgomp 的打包方式。

有意改动的部分：

| 项 | 官方 | 本脚本 |
| --- | --- | --- |
| `TORCH_CUDA_ARCH_LIST`（torch） | `8.0;9.0;10.0;11.0;12.0` | `10.0` |
| `TORCH_CUDA_ARCH_LIST`（torchvision） | `8.0;9.0;10.0;11.0;12.0+PTX` | `10.0` |
| NCCL | 2.29.7，RPATH 为 `$ORIGIN/../../nvidia/nccl/lib` | 2.32.3，RPATH 为 `nccl-2.32.3/build/lib` 的绝对路径 |
| `nvidia-nccl-cu13` 依赖 | 有 | 去掉（否则 `import torch` 时会先预加载它） |
| 平台标签 | `manylinux_2_28_aarch64` | `linux_aarch64`（针对本机 glibc 编译） |

### 什么是 PTX

`TORCH_CUDA_ARCH_LIST` 里每一项（例如 `10.0`）表示为该架构生成 **SASS**，也就是只能在这个架构上运行的机器码。
加上 `+PTX`（例如 `12.0+PTX`）会额外嵌入 **PTX** 中间代码，遇到更新的 GPU 时，驱动可以在运行时把它 JIT 编译成机器码，
用来做前向兼容。官方只在 nightly 版本里给 12.0 加 PTX；正式版（版本号不带 `dev`）不带 PTX，以减小 `libtorch_cuda.so`
（见 `build_env_setup.py` 的 `_ptx_arches()`）。本脚本只编 `10.0` 的 SASS，因此只能在计算能力 10.0 的 GPU 上运行。
