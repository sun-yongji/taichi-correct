# TaiChi-Correct 太极·C6对称误差校正引擎

CCF开源创新大赛2026 · 「太极矩阵」M5模块

## 核心创新

利用C6六重对称耦合矩阵，在六专家/六模态之间检测和校正预测误差。

**三层机制：**
1. **C6共识计算** → 加权耦合矩阵聚合邻居信息
2. **异常检测** → Z-score + Mahalanobis距离在C6耦合空间定位离群专家
3. **迭代精炼** → 异常专家向C6共识收束，非异常专家微调

**对称本征模分解：**
残差在C6本征基上分解为6个模态：
- **uniform_bias** (λ=1.97) — 全局偏置/共同漂移
- **dipole_I/II** (λ=1.50) — 对偶振荡
- **quadrupole_I/II** (λ=0.50) — 四极模态
- **hexapole_noise** (λ=0.03) — 噪声底线

## 安装

```bash
pip install -e .
```

## 使用

```python
from taichi_correct import TaiChiCorrector
import numpy as np

# 6 expert predictions
predictions = np.random.randn(100, 6)
# Expert 2 is noisy
predictions[:, 2] += np.random.randn(100) * 2.0

corrector = TaiChiCorrector(threshold=2.0, max_iter=5)
report = corrector.correct(predictions)
print(f"Confidence: {report.confidence.mean()}")
print(f"Anomaly rate: {report.anomaly_mask.mean()}")
print(f"Iterations: {report.iterations}")
```

## 许可

Apache 2.0
