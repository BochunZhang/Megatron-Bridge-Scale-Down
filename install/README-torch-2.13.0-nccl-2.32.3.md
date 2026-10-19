# GB200：用 NCCL 2.32.3 源码编译 torch 2.13.0 + torchvision 0.28.0

[install-torch-2.13.0-nccl-2.32.3.sh](install-torch-2.13.0-nccl-2.32.3.sh) 按官方 aarch64 cu13 wheel 的编译选项
从源码编 torch 2.13.0 和 torchvision 0.28.0，NCCL **动态链接**到
[install-nccl-2.32.3.sh](install-nccl-2.32.3.sh) 编出来的 `nccl/nccl-2.32.3/build`，只编 GB200（sm_100）。

官方 torch 2.13.0 编译时用的是 NCCL 2.29.7（`nvidia-nccl-cu13==2.29.7`），运行时也加载这份。编完后 torch 的
编译期和运行期 NCCL 都是 2.32.3，并且进程里只有这一份 `libnccl`。

## 安装依赖

在 GB200 节点（Linux aarch64）上运行。下面分成**需要自己装的**和**脚本自动装的**两类。

### 需要自己装的

| 依赖 | 用途 | 检查命令 |
| --- | --- | --- |
| CUDA toolkit 13.0 / 13.2 | 编译 torch / torchvision（nvcc、CUDA 头文件）。官方用 13.0.2 / 13.2.1，小版本不同时脚本会警告 | `/usr/local/cuda/bin/nvcc --version` |
| gcc / g++（官方用 13）、make | C/C++ 编译；版本不是 13 时脚本会警告 | `g++ --version` |
| git、wget、xz、binutils（`readelf`） | 克隆源码；下载并解压 NVPL；检查 `libtorch_cuda.so` 的 NEEDED / RPATH | `git --version; wget --version; readelf -v` |
| Python 3.12（带 `venv` 模块） | 目标 venv 和 torch 的构建 venv | `python3.12 -m venv --help` |
| uv | 脚本里所有 `uv pip` | `uv --version` |
| libpng、libjpeg-turbo、libwebp 开发包 | torchvision 的 PNG / JPEG / WebP 编解码（可选，缺少时编成不带对应编解码的版本，脚本会警告） | `libpng-config --version; ls /usr/include/jpeglib.h /usr/include/webp/decode.h` |
| NCCL 2.32.3 | torch 编译和运行时链接的 NCCL | `bash install/install-nccl-2.32.3.sh --check-only` |

**有 root 时**，一次装齐系统包：

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y git wget xz-utils binutils build-essential \
    python3.12 python3.12-venv python3.12-dev \
    libpng-dev libjpeg-turbo8-dev libwebp-dev        # Debian 上 JPEG 包名是 libjpeg62-turbo-dev

# RHEL / Rocky / AlmaLinux
sudo dnf install -y git wget xz binutils gcc gcc-c++ make \
    python3.12 python3.12-devel \
    libpng-devel libjpeg-turbo-devel libwebp-devel
```

**没有 root 时**，图像编解码库可以用 conda 装到一个目录（官方 torchvision 打包也是这么做的，见
`packaging/pre_build_script.sh`），编译前把路径导出给 torchvision：

```bash
conda create -y -p $HOME/tv-codecs -c conda-forge libpng libjpeg-turbo "libwebp>=1.3.2"
export PATH=$HOME/tv-codecs/bin:$PATH                  # 让 setup.py 找到 libpng-config
export TORCHVISION_INCLUDE=$HOME/tv-codecs/include     # jpeglib.h / webp/decode.h
export TORCHVISION_LIBRARY=$HOME/tv-codecs/lib
```

这种方式装的库在运行时还要设置 `LD_LIBRARY_PATH=$HOME/tv-codecs/lib`，详见下面的
[PNG / JPEG / WebP](#png--jpeg--webp)。

**uv**（没装时任选一种）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # 装到 ~/.local/bin
# 或者
pipx install uv
```

**NCCL 2.32.3**：用 [install-nccl-2.32.3.sh](install-nccl-2.32.3.sh) 编到 `nccl/nccl-2.32.3/build`，见下面运行步骤的第 1 步。

### 脚本自动装的

下面这些不用手动装，脚本会放进 `.tmp/` 或目标 venv：

| 依赖 | 装到哪里 | 说明 |
| --- | --- | --- |
| torch 构建依赖：cmake、ninja、setuptools、numpy 2.1.0 等（`requirements-build.txt`、`requirements.txt`）以及 build、wheel、patchelf | `.tmp/torch-2.13.0-build-py3.12` | 官方版本，只在构建 venv 里，不改动目标 venv |
| cuDNN 9.20.0.48、cuSPARSELt 0.8.1、NVSHMEM 3.4.5、cuda-toolkit 13.0.3 的 pip wheel | 构建 venv（编译用头文件）；目标 venv（作为 torch 的运行时依赖） | 官方固定版本 |
| NVPL（BLAS / LAPACK） | `.tmp/torch-2.13.0-build-deps/nvpl` | `/usr/local/lib` 下没有时，用官方 `install_nvpl.sh` 下载 |
| ACL（Arm Compute Library）和 scons | `.tmp/torch-2.13.0-build-deps/acl` | `/acl` 下没有时，用官方 `install_acl.sh` 从源码编译，比较耗时 |
| build、wheel、packaging、ninja、pyyaml；没有 setuptools 时装 `setuptools<80` | 目标 venv | 编 torchvision 用；只补装缺少的，已有的版本不改 |
| triton 3.7.1 等 torch 运行时依赖 | 目标 venv | 安装 torch wheel 时由 uv 按 wheel 的依赖安装 |

另外，torch 源码（含子模块）、build 目录、NVPL / ACL 都放在 `.tmp/` 下，需要预留几十 GB 磁盘。

## 运行步骤

在 `megatron-bridge` 根目录执行：

```bash
# 0. 装依赖：见上面的“安装依赖”，例如 Ubuntu 上
#    sudo apt-get install -y git wget xz-utils binutils build-essential python3.12 python3.12-venv python3.12-dev \
#        libpng-dev libjpeg-turbo8-dev libwebp-dev

# 1. 编 NCCL 2.32.3 到 nccl/nccl-2.32.3/build，并检查 TMA / multicast / CE 都编进去了
bash install/install-nccl-2.32.3.sh

# 2. 新建并激活一个 Python 3.12 venv（torch 会装进这个 venv）
python3.12 -m venv .venv/python3.12-torch2.13-nccl2.32.3
source .venv/python3.12-torch2.13-nccl2.32.3/bin/activate

# 3. 编 torch，装进当前 venv；再用当前 venv 编 torchvision 并安装；最后跑检查
bash install/install-torch-2.13.0-nccl-2.32.3.sh 2>&1 | tee .tmp/install-torch-nccl-2.32.3.log
```

第 3 步里两个包用的 Python 环境不同：

- **torch** 在 `.tmp/torch-2.13.0-build-py3.12` 这个构建 venv 里编。官方的 torch 构建依赖（`requirements-build.txt`、
  固定的 numpy 等）只装在这里，不会改动你的 venv。
- **torchvision** 用**当前激活的 venv**（`--python`）编：脚本先把刚编好的 torch 装进这个 venv，再用这个 venv 里的
  Python、setuptools 和 torch 执行 `python -m build --no-isolation`，编完再装回这个 venv。脚本只会补装缺少的
  `build` / `wheel` / `packaging` / `ninja` / `pyyaml`；venv 里没有 setuptools 时装 `setuptools<80`，已有的版本都不改。

第 3 步的输出：

| 输出 | 位置 |
| --- | --- |
| torch / torchvision wheel | `.tmp/wheels/torch-2.13.0-nccl-2.32.3/torch-2.13.0+cu130-cp312-cp312-linux_aarch64.whl`、`torchvision-0.28.0+cu130-...whl` |
| 编译日志 | `.tmp/torch-build.log`、`.tmp/torchvision-build.log` |
| torch 的编译用 venv（官方的编译依赖，不污染目标 venv） | `.tmp/torch-2.13.0-build-py3.12` |
| 官方编译环境 / 运行时依赖 | `.tmp/torch-2.13.0-build.env`、`.tmp/torch-2.13.0-runtime-requirements.txt` |

成功时最后一行是 `=== torch 2.13.0+cu130 + torchvision ready in ...`，有检查失败时脚本以非 0 退出，并打印失败项数量。

### 常用参数

| 参数 | 作用 |
| --- | --- |
| `--python PY` | 目标环境：torch / torchvision 装进这里，torchvision 也用它编译。默认当前 venv 的 `python` |
| `--nccl-home DIR` | NCCL 2.32.3 的 build 目录，默认 `nccl/nccl-2.32.3/build` |
| `--cuda-home DIR` | CUDA toolkit，默认 `CUDA_HOME` 或 `/usr/local/cuda` |
| `--jobs N` | 编译并发数 `MAX_JOBS`，默认按官方取 `min(nproc - 2, 12)` |
| `--skip-torch-build` | 复用 wheel 目录里已编好的 torch wheel：装进 `--python`，再重编 torchvision |
| `--no-install` | 只编 torch wheel，不安装（torchvision 需要 torch 先装进 `--python`，所以这时不编） |
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

## PNG / JPEG / WebP

torchvision 的 `setup.py` 编译时按下面的方式查找图像编解码库，找不到就跳过对应功能（`torchvision.io` 的
`decode_png` / `decode_jpeg` / `decode_webp` 等不可用），不会报错：

| 库 | 查找方式 |
| --- | --- |
| PNG（libpng ≥ 1.6） | `PATH` 里的 `libpng-config` |
| JPEG（libjpeg-turbo） | `jpeglib.h`：依次在 `TORCHVISION_INCLUDE`、`$CONDA_PREFIX/include`、`/usr/include`、`/usr/local/include` 中找 |
| WebP（libwebp ≥ 1.3.2） | `webp/decode.h`：同上 |

安装命令见上面的[安装依赖](#安装依赖)。依赖装好后，torch 不用重编，只重编 torchvision：

```bash
bash install/install-torch-2.13.0-nccl-2.32.3.sh --skip-torch-build
```

脚本编 torchvision 时会保留 `TORCHVISION_INCLUDE` / `TORCHVISION_LIBRARY`。注意：编出来的 torchvision **动态链接**这些库，
官方 wheel 会把它们打包进 wheel，本脚本不会。用系统包时运行时自然能找到；用 conda 装到 `$HOME/tv-codecs` 时，运行时要设置
`LD_LIBRARY_PATH=$HOME/tv-codecs/lib`，否则 `import torchvision` 会报找不到 `libpng16.so` / `libjpeg.so` / `libwebp.so`。

确认是否编进去了：

```bash
grep -E "Building torchvision with (PNG|JPEG|WEBP)|without" .tmp/torchvision-build.log
python -c "import torch, torchvision.io as io; x = torch.zeros(3, 8, 8, dtype=torch.uint8); io.decode_png(io.encode_png(x)); io.decode_jpeg(io.encode_jpeg(x)); print('png/jpeg ok')"
```

## 装到其他 venv

wheel 编好后，其他 venv 不需要重新编译：

```bash
python3.12 -m venv .venv/another-env && source .venv/another-env/bin/activate
W=.tmp/wheels/torch-2.13.0-nccl-2.32.3
# UV_NO_CONFIG=1 必须加：见下面“注意事项”第一条
UV_NO_CONFIG=1 uv pip install $W/torch-2.13.0+cu130-*.whl $W/torchvision-0.28.0+cu130-*.whl
bash install/install-torch-2.13.0-nccl-2.32.3.sh --check-only
```

## 注意事项

- **在仓库目录下手动 `uv pip install` 这两个 wheel 时，要加 `UV_NO_CONFIG=1`。** `uv pip` 会读取当前目录
  `pyproject.toml` 的 `[tool.uv]`，而 megatron-bridge 的 `override-dependencies` 把 torch / torchvision 改写成了
  `sys_platform == 'never'`，uv 会直接丢掉这两个要求，只输出 `Resolved in ...ms`，不报错也不安装。脚本内部已经设置了这个变量。
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
