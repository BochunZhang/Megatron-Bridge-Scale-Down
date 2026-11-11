git submodule update --init --remote 3rdparty/Megatron-LM

uv pip install --no-build-isolation --no-deps --no-cache --reinstall 3rdparty/Megatron-LM/src/nvme/deepspeed
