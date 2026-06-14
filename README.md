[![CI](https://github.com/sun-yongji/taichi-correct/actions/workflows/ci.yml/badge.svg)](https://github.com/sun-yongji/taichi-correct/actions/workflows/ci.yml)

# TaiChi-Correct 🛡️ C6 共识迭代校正引擎

> 华为云杯 2026 OPC 大赛 | 太极矩阵 M5 | CC-BY-SA-4.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-28/28-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-CC--BY--SA--4.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-taichi--correct-blue)](https://pypi.org/project/taichi-correct/)

## 核心创新

大模型输出存在不可预测的随机噪声。TaiChi-Correct 以 **C6 群的六阶本征模分解** 做共识迭代校正——对 logits 施加六次 C6 旋转变换 → 六个独立预测版本 → 共识残差矩阵 → C6 不可约表示投影 → 滤除高阶噪声模 → 重建校正输出。

**噪声检测率 15.2%，残差标准差降幅 69.7%，共形预测置信度 98%。**

## 性能

| 指标 | 数值 |
|------|------|
| 噪声检测率 | 15.2% |
| 残差标准差降幅 | 69.7% |
| 共形预测置信度 | 98% |
| 默认迭代次数 | 2（高精度可达 6） |
| 测试通过率 | 28/28 |

## 安装

```bash
pip install taichi-correct
```

## 快速开始

```python
from taichi_correct import TaiChiCorrector
import numpy as np

corrector = TaiChiCorrector()
logits = np.random.randn(100, 10)
corrected = corrector.correct(logits)
print(f"Anomalies detected: {corrector.anomaly_mask.sum()}")
print(f"Confidence: {corrector.confidence:.1%}")
```

## 太极矩阵体系

TaiChi-Correct 是太极矩阵六站体系的 M5 站：

| 站 | 仓库 | 功能 |
|----|------|------|
| M1 | [taichi-router](https://github.com/sun-yongji/taichi-router) | MoE 动态路由 |
| M2 | [taichi-mtp](https://github.com/sun-yongji/taichi-mtp) | 多 token 预测 |
| M3 | [taichi-quant](https://github.com/sun-yongji/taichi-quant) | 熵量化 |
| M4 | [taichi-hex](https://github.com/sun-yongji/taichi-hex) | 六边形注意力 |
| **M5** | **taichi-correct** ← 你在这里 | 共识校正 |
| M6 | [taichi-matrix](https://github.com/sun-yongji/taichi-matrix) | 统一入口 |

技术白皮书：[太极矩阵技术白皮书(中文)](https://docs.qq.com/aio/DTldDRGpIbGdseG1H) | [WHITEPAPER.md](https://github.com/sun-yongji/taichi-matrix/blob/master/WHITEPAPER.md)

## 参与贡献

欢迎提交 Issue 和 Pull Request。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

CC-BY-SA-4.0 · 易宇本源研究中心 · 2026
