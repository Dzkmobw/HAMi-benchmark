# HAMi Benchmark

这个仓库用于比较三种运行模式在单卡混合负载下的表现：

- `HAMi only`
- `HAMi + external scheduler`
- `host-direct baseline`

它不是通用机器学习 benchmark，而是一个调度评估仓库。
关注的核心指标是：

- 平均任务完成时间
- GPU 利用率
- 显存利用率

一、怎么用

运行前默认假设：

- HAMi 源码位于：
  - `/home/ttk/project/HAMi`
- external scheduler 位于：
  - `/home/ttk/project/HAMi-external-scheduler`
- 当前仓库位于：
  - `/home/ttk/project/HAMi-benchmark`

如果要跑 K8s 模式，要求：

- `minikube` 可用
- HAMi 已启动
- 如果跑 external 模式，external controller 已启动

运行 HAMi-only：

```bash
/home/ttk/project/HAMi-benchmark/mini/run-mini-hami-only
```

运行 external：

```bash
/home/ttk/project/HAMi-benchmark/mini/run-mini-external
```

清理 benchmark 资源：

```bash
/home/ttk/project/HAMi-benchmark/mini/cleanup-mini
```

运行宿主机 baseline：

```bash
/home/ttk/project/HAMi-benchmark/host-mini/run-host-mini
```

二、项目有什么功能

这个仓库主要提供四类能力。

第一类是 benchmark 计划定义。
在这里：

- [`benchmark-plan.yaml`](/home/ttk/project/HAMi-benchmark/benchmark-plan.yaml)
- [`benchmark-plan-mini.yaml`](/home/ttk/project/HAMi-benchmark/benchmark-plan-mini.yaml)

它们负责描述测试目标、对比模式和负载设计思路。

第二类是可运行的 K8s benchmark。
在这里：

- [`mini/`](/home/ttk/project/HAMi-benchmark/mini)

这一部分会真正提交 Job、采样指标并生成 summary。

第三类是宿主机 baseline。
在这里：

- [`host-mini/`](/home/ttk/project/HAMi-benchmark/host-mini)

它用于提供一个不经过 K8s、不经过 HAMi 的参考基线。

第四类是自动日志和结果汇总。
每次 benchmark 结束后，都会自动生成：

- `run.log`
- `metrics_samples.csv`
- `summary.json`
- `summary.txt`

在 K8s 模式下通常还会有：

- `pods.json`
- `jobs.json`

在 host 模式下通常会有：

- `tasks.json`

三、目录和模块说明

[`benchmark-plan.yaml`](/home/ttk/project/HAMi-benchmark/benchmark-plan.yaml)

- 完整 round 的概念设计文件
- 更偏设计说明，不是直接运行入口

[`benchmark-plan-mini.yaml`](/home/ttk/project/HAMi-benchmark/benchmark-plan-mini.yaml)

- mini round 的概念设计文件
- 用来描述快速验证方案

[`mini/`](/home/ttk/project/HAMi-benchmark/mini)

- 主 K8s benchmark 目录
- 包含 manifests、运行脚本、采样与汇总脚本

[`mini/workloads/mini_workload.py`](/home/ttk/project/HAMi-benchmark/mini/workloads/mini_workload.py)

- 所有 mini 任务共用的 Python workload 入口
- benchmark 真正执行的 GPU workload 在这里

[`mini/manifests/hami-only`](/home/ttk/project/HAMi-benchmark/mini/manifests/hami-only)

- plain HAMi 模式的 Job manifests

[`mini/manifests/external`](/home/ttk/project/HAMi-benchmark/mini/manifests/external)

- external scheduler 模式的 Job manifests

[`mini/run-mini-common`](/home/ttk/project/HAMi-benchmark/mini/run-mini-common)

- 两种 K8s 模式共用的执行器
- 负责建 namespace、刷新 ConfigMap、提交流程、等待完成、写 summary

[`mini/run-mini-hami-only`](/home/ttk/project/HAMi-benchmark/mini/run-mini-hami-only)

- HAMi-only 入口

[`mini/run-mini-external`](/home/ttk/project/HAMi-benchmark/mini/run-mini-external)

- external 模式入口

[`mini/cleanup-mini`](/home/ttk/project/HAMi-benchmark/mini/cleanup-mini)

- 清理 mini benchmark 资源

[`mini/tools/benchmark_logger.py`](/home/ttk/project/HAMi-benchmark/mini/tools/benchmark_logger.py)

- 负责指标采样和 summary 汇总
- 这是结果统计的关键文件

[`host-mini/`](/home/ttk/project/HAMi-benchmark/host-mini)

- 宿主机 baseline 目录

[`host-mini/tools/host_benchmark_runner.py`](/home/ttk/project/HAMi-benchmark/host-mini/tools/host_benchmark_runner.py)

- host baseline 的主执行器
- 用 `docker run --gpus all` 跑任务
- 用 `nvidia-smi` 采样 GPU 指标

[`host-mini/run-host-mini`](/home/ttk/project/HAMi-benchmark/host-mini/run-host-mini)

- 宿主机 baseline 的入口脚本

四、日志输出在哪里

K8s mini benchmark 的日志目录在：

- [`mini/logs`](/home/ttk/project/HAMi-benchmark/mini/logs)

宿主机 baseline 的日志目录在：

- [`host-mini/logs`](/home/ttk/project/HAMi-benchmark/host-mini/logs)

每一轮 benchmark 都会生成一个带时间戳的目录。
最常看的通常是：

- `summary.txt`
- `summary.json`
- `run.log`
- `metrics_samples.csv`

五、当前已知注意事项

第一，概念 plan 和当前 runnable 版本不一定完全一致。
如果你在改 benchmark，真正应该优先看的不是高层计划说明，而是：

- `mini/manifests/`
- `host-mini/tools/host_benchmark_runner.py`

第二，结果会明显受 HAMi 的 node lock 配置影响。
当前默认假设是 HAMi 以 `30s` 的 node lock timeout 运行。

第三，external scheduler 是否“工作正常”和它是否“比 HAMi-only 更强”不是一回事。
这个仓库既用于验证链路能不能跑，也用于验证策略到底有没有带来收益。

六、推荐阅读顺序

如果第一次接手这个仓库，建议按下面顺序读：

1. `README.md`
2. `benchmark-plan.yaml`
3. `mini/run-mini-common`
4. `mini/tools/benchmark_logger.py`
5. `mini/workloads/mini_workload.py`
6. `host-mini/tools/host_benchmark_runner.py`

这样最快能理解：

- benchmark 目标
- benchmark 怎么跑
- 指标是怎么采样和汇总的

七、最常用命令

清理：

```bash
/home/ttk/project/HAMi-benchmark/mini/cleanup-mini
```

跑 HAMi-only：

```bash
/home/ttk/project/HAMi-benchmark/mini/run-mini-hami-only
```

跑 external：

```bash
/home/ttk/project/HAMi-benchmark/mini/run-mini-external
```

跑 host baseline：

```bash
/home/ttk/project/HAMi-benchmark/host-mini/run-host-mini
```
