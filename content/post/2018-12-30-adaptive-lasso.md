---
title: adaptive lasso
author: 李智凡
date: '2018-12-30'
slug: adaptive-lasso
categories:
  - statistics
tags:
  - lasso
---

# The Adaptive Lasso and its oracle properties

* 简述基本想法
* 主要内容

## introduction

统计模型学习有两个最基本的目标：保证好的预测准确性；筛选出相关的有预测价值的变量。特别是当真模型有稀疏表示时，变量选择尤为重要。确定有意义的预测变量，当然可以加强模型的预测表现。

沿用Fan和Li(2001)的描述：我们考虑一下线性回归模型下，变量选择和模型估计的问题。假设$y = (y_1,y_2,\cdots,y_n)$为响应，$x_j = (x_{1j},x_{2j},\cdots,x_{nj}),j =1,2,\cdots,p$是线性无关的预测变量，设$X = [x_1,x_2,\cdots,x_p]$为预测变量矩阵，不失一般性，我们假定数据是中心化的。假设真模型的参数集合$\mathcal{A} = \{j:\beta^{*}_{j} \ne 0 \}$，且$|\mathcal{A}| = p_0 < p $，换言之，真模型中预测变量是已知所有预测变量的真子集。我们设通过估计过程$\delta$得到的参数估计为$\hat{\beta}(\delta)$，那么，我们称这个过程是一个**Oracle procedure**:

* 能得到正确的子模型：$\{j:\hat{\beta}_{j} \ne 0\}=\mathcal{A}$
* 参数估计有：$\sqrt{n}(\hat{\beta}(\delta)_{\mathcal{A}} - \beta^{*}_{\mathcal{A}})\stackrel{d}{\longrightarrow}N(0,\Sigma^{*})$，其中$\Sigma^{*}$真子模型的协方差矩阵。

一般的最小二乘估计(OLS)无法进行变量选择，于是，传统的做法是采用最优子集回归，但有两个明显缺陷：

1. 当预测变量个数很大时，选择最优子集计算上很难实现；
2. 通常采用stepwise的方法，容易收敛到局部最优解。

Lasso是同时进行变量选择和模型估计的正则化技术。它的估计可以定义如下：
$$
\hat{\beta}(lasso) = \arg \min_{\beta}\parallel y-\sum_{i=1}^{p}x_j \beta_j\parallel+\lambda \sum_{j=1}^{p}|\beta_j|
$$
其中$\lambda$为非负的正则化参数，第二项称为**$l_1$penalty**。关于lasso的估计是否是一个oracle procedure是一个重要问题。本文将给出这个问题的答案。特别的，我们感兴趣的是$l_{1}$penalty是否可以成为一个oracle procedure，如果是，需要满足什么条件。我们考虑一个近似的设定，是关于$\lambda_{n}$随n变化而变化的。首先我们说明，如果lasso的变量选择具有oracle性质，则设定的模型需要满足一个不平凡的假定，换言之，在某些情形下，lasso的变量选择没有oracle性质。

## the lasso variable selection could be inconsistent

首先，沿用Knight与Fu(2000)中的近似理论分析：

* $y_i = x_i \beta^{*} + \epsilon_{i}$，其中$\epsilon_i$是独立同分布随机变量，均值为0，方差为$\sigma^2$。
* $\frac{1}{n}X^{T}X \to C$，其中$C$为正定矩阵。
* 假设真参数集合$\mathcal{A} = \{1,2,\cdots,p_0\}$，令：

$$
C = \left[\begin{array}{cc}
C_{11} & C_{12} \\
C_{21} & C_{22}
\end{array}\right]
$$

其中$C_11$为$p_0 \times p_0$的矩阵。

考虑lasso的估计问题如下：
$$
\hat{\beta}^{(n)} =  \arg \min_{\beta}\parallel y-\sum_{i=1}^{p}x_j \beta_j\parallel+\lambda_{n} \sum_{j=1}^{p}|\beta_j| \tag{2}
$$
其中，$\lambda_{n}$随$n$变化而变化，设$\mathcal{A} = \{j:\hat{\beta}^{(n)}\ne 0\}$，则lasso的变量选择具有一致性，当且仅当$\lim_{n}  P(\mathcal{A}_n - \mathcal{A})=1$。则有如下两个引理：

**lemma 1**

如果$\lambda_n/n \to \lambda_{0} \ge 0$，则$\hat{\beta}^{(n)}\stackrel{p}{\longrightarrow} \arg \min V_1$,其中$V_1$为：
$$
V_1(u) = (u-\beta^{*})^{\Tau}C(U-\beta^{*})+\lambda_0\sum_{j=1}^{p}|u_j|
$$


**lemma 2**

如果$\lambda_n /\sqrt{n} \to \lambda_0 \ge 0$，则$\sqrt{n}(\hat{\beta}^{(n)}-\beta^{*}) \stackrel{d}\to \arg\min(V_2)$，其中：
$$
V_2(u) = -2u^{\Tau}W+u^{\Tau}Cu+\lambda_0\sum_{j=1}^{p}[u_j sgn(\beta_{j}^{*})I(\beta_{j}^{*}\ne 0)+|u_j|I(\beta^{*}_{j} = 0)]
$$
且$W\sim N(0,\sigma^2 C)$

引理1说明对于参数估计来说，只有当$\lambda_0 = 0$是，才有参数估计的一致性。然而，我们如果考虑变量选择，引理2得出，当$\lambda_n  = O(\sqrt{n})$，$\mathcal{A_n} $有一个正概率不等于$\mathcal{A}$。即命题1：

**Proposition 1**

如果$\lambda_n /\sqrt{n} \to \lambda_0 \ge 0$，则$\lim \sup_{n}P(\mathcal{A}_n = \mathcal{A}) \le c <1$，其中$c$为一个正常数，由真模型决定。

**lemma 3**

由命题1，可以衍生出$\lambda_0 = \infty$的情况，从而，我们考虑$\lambda_n/n \to  0 , \lambda_{n} /\sqrt{n} \to \infty$，则有$\frac{n}{\lambda_n}(\hat{\beta}^{(n)}-\beta^{*})\stackrel{p}\to\arg \min(V_3)$：
$$
V_3(u) = u^{\Tau}Cu+\sum_{j=1}^{p}[u_j sgn(\beta_{j}^{*})I(\beta_{j}^{*}\ne 0)+|u_j|I(\beta^{*}_{j} = 0)]
$$


由lemma 3可以看出，此时$\hat{\beta}^{(n)}$的收敛速度比$\sqrt{n}$慢。因此，如果要有良好的收敛性，只能当$\lambda_{n} = 0(\sqrt{n})$，但此时，变量选则没有相合性。

接下来，我们需要思考，如果我们牺牲一下参数收敛速率，能使得变量选择的相合性是成立的。事实上，这一点是没有办法保证的。接下来的定理给出了变量选择相合性的一个必要条件：

**定理 1**

假设$\lim_{n}P(\mathcal{A}_n = \mathcal{A}) = 1$，则存在某个示性向量$s = (s_1,s_2,\cdots,s_{p_0}),s_j = \pm 1$，使得：
$$
|C_{21}C_{11}^{-1}s| \le 1 \tag{3}\label{eq:3}
$$
这个不等式指各分量需要绝对值不大于1.

事实上，这个条件不是平凡的，我们可以构造一个如下例子：

**推论 1**

假设$p_0 = 2m+1 \ge 3$，且$p = p_0+1$，意味着有一个变量为无关变量。假设$C_{11} = (1-\rho_{1})I+\rho_{1}J_1$，其中$J_1$为元素为1的矩阵；且$C_{12} = \rho\vec{1},C_{22}=1$，如果$-\frac{1}{p_0-1}< \rho_{1} < -\frac{1}{p_0}, 1+ (p_0-1)\rho_1 < |\rho_2|<\sqrt{(1+(p_0-1)\rho_1)/p_0}$，则此时条件$\eqref{eq:3}$不满足，即此时变量选择没有相合性。