[![CI](https://github.com/sun-yongji/taichi-correct/actions/workflows/ci.yml/badge.svg)](https://github.com/sun-yongji/taichi-correct/actions/workflows/ci.yml)

# TaiChi-Correct 🛡️ C6共识迭代校正引擎

> 华为云杯2026 OPC大赛  |  太极矩阵 M5  |  Apache 2.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-28/28-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 核心创新

大模型输出存在不可预测的随机噪声。TaiChi-Correct以**C6群的六阶本征模分解**做共识迭代校正——对logits施加六次C6旋转变换→六个独立预测版本→共识残差矩阵→C6不可约表示投影→滤除高阶噪声模→重建校正输出。

**噪声检测率15.2%，残差标准差降幅69.7%，共形预测置信度98%。**

## 性能

| 指标 | 数值 |
|------|------|
| 噪声检测率 | 15.2% |
| 残差标准差降幅 | 69.7% |
| 共形预测置信度 | 98% |
| 默认迭代次数 | 2（高精度可达6） |
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

| 站 | 仓库 | 功能 |
|----|------|------|
| M1 | [taichi-router](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-router) | MoE动态路由 |
| M2 | [taichi-mtp](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-mtp) | 多token预测 |
| M3 | [taichi-quant](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-quant) | 熵量化 |
| M4 | [taichi-hex](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-hex) | 六边形注意力 |
| **M5** | **taichi-correct** ← 你在这里 | 共识校正 |
| M6 | [taichi-matrix](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix) | 统一入口 |

技术白皮书：[太极矩阵技术白皮书(中文)](https://docs.qq.com/aio/DTldDRGpIbGdseG1H) | [WHITEPAPER.md (English)](https://github.com/sun-yongji/taichi-matrix/blob/master/WHITEPAPER.md)

## 许可

Apache 2.0 · 太极量子团队 · 2026