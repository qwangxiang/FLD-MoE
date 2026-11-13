# FLD-MoE

模型由 5 个独立双向 TCN 专家和 XGBoost 门控组成，在 902 个等距价格节点上直接输出完整概率质量。专家与门控通过预测损失产生的软责任交替训练。

## 安装与一键运行

需要 Python 3.12+ 和 `uv`。在仓库根目录执行：

```bash
uv sync --locked
uv run --locked fld-moe train
```

训练命令完成数据准备、训练、恢复验证集上最优的同轮专家与门控、预测验证集和测试集、计算并保存指标。默认自动选择 CUDA 或 CPU；终端显示逐轮训练进度和最终测试指标。依赖版本由 `uv.lock` 固定。

```bash
# 显式选择设备及一个尚不存在的输出目录
uv run --locked fld-moe train --device cuda --output results/shaanxi-run-01

# 等价的 Python 模块入口
uv run --locked python -m fld_moe train
```

`train` 使用全部训练样本及固定参数，最多 600 轮。默认保护期为 76 轮，之后根据验证集尺度归一 CRPS 早停。

## 只读数据检查

```bash
# 只读取数据、构造特征并报告切分，不训练
uv run --locked fld-moe check-data
```

该命令读取数据、构造特征并报告切分，不执行训练。

## 数据与切分

`data/shaanxi.csv` 包含 2025-01-01 至 2026-04-30 的陕西数据，每天 96 个 15 分钟点。详细列说明见 [data/README.md](data/README.md)。价格单位为元/MWh。


| 切分 | 目标日期                 | 目标日数 |
| ------ | -------------------------- | ---------: |
| 训练 | 2025-01-29 至 2026-02-14 |      382 |
| 验证 | 2026-02-15 至 2026-03-16 |       30 |
| 测试 | 2026-03-17 至 2026-04-30 |       45 |

原始训练区间为前 410 天，其中前 28 天用于历史暖机。专家输入 24 列原始外生变量和 12 列时间编码（原 `raw` 档）；门控直接使用陕西 `super_abundant` 档全部 1107 列特征，包括原始变量、时间编码、同刻滞后、历史价格统计、动量、波动率、均值回复与跨列衍生特征。列顺序由特征工程代码生成，不读取筛选快照。

门控采用完整特征，其他有效超参数沿用陕西配置，未针对这一输入变化重新调参。

行政限价为 2025 年 `[40, 1000]`、2026 年 `[0, 1000]`。历史价格先按其观测当时的限价归一化，历史统计和标准化均仅使用目标日之前的信息。验证与测试按日滚动预测，可读取此前已经发生的价格；模型和门控不在这两个切分上重新拟合。完整目标日的外生预测变量用于非因果 TCN 的日内上下文建模。

参数和数据路径均在 [configs/shaanxi.toml](configs/shaanxi.toml) 中。路径相对于配置文件解析，支持 `--config PATH`。实现仅覆盖陕西配置使用的两条特征流及完整自然日训练，不包含特征筛选器、超参数搜索及其他市场或模型。

## 结果与指标

每次运行默认生成 `results/train-<UTC时间>/`；显式指定的目录必须尚不存在。


| 文件                                         | 内容                                                 |
| ---------------------------------------------- | ------------------------------------------------------ |
| `metrics.json`                               | 验证和测试指标、单位及评价阈值                       |
| `test_predictions.csv`                       | 真实价格、中位数、10%/90% 分位及逐点评分             |
| `test_distribution.npz`                      | 完整原生价格网格、PMF、时间戳和真实价格              |
| `validation_*`                               | 同样格式的验证预测                                   |
| `experts.pt`、`gate.ubj`                     | 同一最优轮的专家权重和树门控                         |
| `history.json`                               | 各轮训练记录、最佳轮次和门控内部切分                 |
| `input_config.toml`、`effective_config.json` | 原始输入配置与实际生效配置（包含设备覆盖）           |
| `data_contract.json`                         | 完整特征顺序、因果归一化规则、日期及价格段初始化信息 |
| `run.json`、`manifest.json`                  | 环境版本、运行时间、输入/源码/产物 SHA256            |

指标包括 CRPS、MAE、RMSE、R²、PICP80、MPIW80、IS80，并额外提供阈值为 400 的 twCRPS。CRPS 直接解析积分原生阶梯 CDF；点预测取原生分布中位数，区间取 10%/90% 分位，PICP80 使用 `[0,1]` 比例。验证早停的尺度归一 CRPS 与最终按物理价格单位报告的 CRPS 区分记录。常量真值的 R² 记为 `null`。

独立复算测试指标不需要重新训练：

```bash
uv run --locked fld-moe evaluate \
  --forecast results/shaanxi-run-01/test_distribution.npz
```

非默认 twCRPS 阈值可通过 `--tail-threshold` 指定；结果可通过 `--output metrics-recomputed.json` 保存到新文件。

## 代码结构

```text
configs/                 陕西固定参数
data/                    陕西原始实验数据
src/fld_moe/
  config.py              参数读取与契约校验
  features.py            专家原始特征和完整陕西门控特征
  transforms.py          当前配置所需的特征变换
  data.py, schema.py     因果数据准备和数据契约
  model.py               独立双向 TCN 专家
  distribution.py        等距价格网格和专家概率混合
  loss.py                完整离散 CRPS 和责任归一损失
  gate.py                软目标 XGBoost 门控
  training.py            保护期、交替更新与配对早停
  evaluation.py          原生分布评价与预测文件读写
  artifacts.py           检查点和运行产物
  api.py, cli.py          流程编排和命令入口
```

代码不依赖原论文目录或本机绝对路径。陕西有效超参数来自原 `PricePredict/FLD_MoE_Uniform/params/shaanxi.toml`；其中两个网格计数字段合并为 `grid_points=902`，零权重的条件尾部损失及未参与等距坐标计算的波动率参数已删除，保留原训练样本的 28 天暖机口径。

完整特征及归一化结果与原实现未筛选模式逐值对齐；已完成临时小规模端到端验证，未在本机执行正式全量训练。

## 许可证

代码和文档使用 [MIT License](LICENSE)。`data/shaanxi.csv` 的许可范围单独说明于 [data/README.md](data/README.md)，不自动继承代码许可证。
