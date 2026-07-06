# PINN-Schrodinger

基于物理信息神经网络（PINN）求解含时薛定谔方程，模拟量子隧穿效应。

## 物理问题

一维含时薛定谔方程（无量纲单位）：

$$i\frac{\partial\psi}{\partial t} = -\frac{\partial^2\psi}{\partial x^2} + V(x,t)\psi$$

势函数（周期性微扰势垒）：

$$V(x,t) = V_0 \cdot \text{sech}^2(x) \cdot [1 + \varepsilon \cdot \sin(\omega t)]$$

初始条件为向右运动的高斯波包。

## 文件说明

| 文件 | 说明 |
|------|------|
| `pinn_run.py` | PINN 求解器：网络定义、自动微分、训练、预测 |
| `traditional_solver.py` | 传统方法：Crank-Nicolson 有限差分 + Split-Step Fourier |
| `visualization.py` | 可视化：波函数快照、热力图、隧穿分析、误差对比 |
| `量子力学PINN实验报告.docx` | 完整实验报告 |

## 运行

```bash
pip install -r requirements.txt

# 先运行传统方法（1-2秒），生成参考解
python traditional_solver.py

# 再运行 PINN（约1-2分钟），训练神经网络
python pinn_run.py

# 生成对比图和动画
python visualization.py
```

## 结果

| 指标 | 数值 |
|------|------|
| PINN 网络结构 | [2, 64, 64, 64, 64, 2] |
| 训练周期 | 5000 epochs |
| 最终 PDE 残差 | 2.81×10⁻³ |
| PINN vs SSF 平均 L² 误差 | 0.3594 |
| 量子隧穿概率 | 49.6% |
| 概率守恒性 | 99.5% |

### 波函数演化

![波函数对比](results/fig_initial_final.png)

### 训练收敛

![训练损失](results/fig_loss.png)

### 量子隧穿

![隧穿分析](results/fig_tunneling.png)
