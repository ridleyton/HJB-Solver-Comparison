# 目标期刊选择 与「距 JCR 一区」差距清单

**对象**：`rework/paper/main.tex`（CISP-BMEI 2026 投稿版，11 页 IEEEtran，6 图 7 表 40 条参考文献）
**目的**：① 选一个 JCR 一区（Q1）期刊；② 把「还差什么」写成可复核的条目，而不是印象。
**复核方式**：本文件每条断言都给出证据位置或复核命令，见附录 A。

---

## 摘要（先看这一页）

**选刊结论**

| 角色 | 期刊 | 依据 |
|---|---|---|
| **主投** | **Journal of Computational Physics**（Elsevier） | JCR Q1（**PHYSICS, MATHEMATICAL，排名 3/60**），JIF 3.9（JCR 2025）；**其投稿指南明确要求"与既有方法比较"**，且 2025 年刚发表过同体裁的 PINN 竞争力测量论文（见 §1.2 证据）。首轮约 2.8 个月 |
| **想 2–4 周内就投出去** | **Applied Mathematics and Computation**（Elsevier） | JCR Q1（**MATHEMATICS, APPLIED**），IF 3.4（JCR 2024）；体裁保守、几乎不需改写。用"稳"换"档次" |
| 备选 | Journal of Scientific Computing（Springer） | JCR Q1（数学应用，37/345），JIF 2.6（JCR 2025），不在任何中科院预警名单 |
| 备选（换视角） | Computers & Operations Research（Elsevier） | JCR Q1（OR & MS，25/109），JIF 4.6；OR 审稿周期最短（约 3.3 个月），但需 OR 化改写 |
| **明确排除** | Quantitative Finance（非 Q1，中科院 4 区）；Omega、Engineering Applications of AI（**范围声明明文排除**）；Mathematical Finance / Finance and Stochastics / SIAM J. Financial Mathematics（理论刊，无定理即拒） | 见 §1.4 |

> **时间基准**：今天是 2026-10-14，最新一轮是 **2026-06-17 发布的 JCR 2025**（JIF 3.9 等为其数值），上一轮是 JCR 2024。本文件同时标注两个年份，因为部分期刊只查到了 2024 数据。**Clarivate 官方站与各出版商官网在本环境被 Cloudflare/JS 拦截，数值来自镜像服务（AbleSci/BioxBio/武大图书馆指南），每格均给出来源；查不到的标 UNVERIFIED，不猜。**

**一句话状态**：这是一篇**完成度很高的会议论文**，不是一篇"投出去等结果"的期刊论文。它的**证据链是 Q1 水准**（精确参考解、可复现、结论诚实），但它的**论证形态是会议水准**（全部结论靠数值观察，没有一个命题），而且**存在两处硬问题**（口径与取行 bug，见下）——这两处必须在自己发现之前先修掉。

**差距排序（按"不做的后果"排）**

| 级别 | 条目 | 后果 |
|---|---|---|
| **P0-A** | **误差口径不统一**：三个表的"收敛阶"和全部头条比值一半是"平方误差"口径 | 审稿人一算就发现数字差 2 倍（阶）或 √ 倍（比值），直接损伤可信度 |
| **P0-A2** | **`tab_bsde` 的 p=4 行取错了维度**：自动生成器选中了 d=8 的记录 | 表里的 0.3% 应为 3.3%，表与正文互相矛盾；好消息是**修正后对本文结论更有利** |
| **P0-B** | 头部主张只有数值证据，**没有命题/证明** | 数值类 Q1 期刊的标准拒稿理由："this is a benchmarking exercise" |
| **P0-C** | 高阶重构**破坏单调性**的问题未讨论（Barles–Souganidis 框架） | 数值分析审稿人必问：高阶插值的梯度凭什么还是粘性解意义下的收敛格式 |
| **P0-D** | 基准问题的**解是二次多项式**，网格法被天然优待（N\*=21 与维度无关） | 直接威胁"等精度下网格更快"这条头部结论的普适性 |
| **P0-E** | 缺关键基线：LSMC（已承认未跑）、高层次概率方法 | "PINN 在高维是唯一能跑的方法"这句话目前只对**一个**弱基线成立 |
| **P0-F** | deep BSDE 报的是 **oracle 选点**（按精确解挑最优 epoch） | 审稿人：这个精度是"知道答案才能拿到"的，不可复现 |
| **P1** | 统计强度（3 种子、重尾 R²、表间 −57 vs −3.28 不一致）、边界/截断域未讨论、文献定位缺 ENO/WENO 与 PDE 最优控制误差估计、真实数据部分偏薄、成本比较的硬件公平性 | 大修意见的主体 |
| **P2** | 期刊模板/Highlights/CRediT/数据声明、Zenodo DOI、图表增补、母语润色 | 编辑层面，不致命 |

---

# 第一部分 · 选刊

## 1.1 候选期刊核对表

**JCR 数据年份**：最新为 **JCR 2025（2026-06-17 发布）**；括号内给出 JCR 2024 数值以便对比。**中科院分区与 JCR 四分位是两套体系，不可混用**（本表分列）。

| 期刊 | 出版商 | JIF | JCR 学科（分区，排名） | 中科院 2025 大类 | 契合度 | 风险 |
|---|---|---|---|---|---|---|
| **Journal of Computational Physics** | Elsevier | **3.9**（2024: 3.8） | PHYSICS, MATHEMATICAL **(Q1, 3/60)**；CS, INTERDISC. APPL. (Q2, 51/169) | 物理与天体物理 **2 区**（Top=是；小类 物理:数学物理 **1 区**） | ★★★★★ | **低**（指南明文要求比较；已有同体裁先例） |
| **Applied Mathematics and Computation** | Elsevier | 3.4（JCR 2024，已核；2025 未核） | MATHEMATICS, APPLIED **(Q1)** | 数学 **2 区** | ★★★★★ | 低 |
| Journal of Scientific Computing | Springer | **2.6**（2024: 3.3） | MATHEMATICS, APPLIED **(Q1, 37/345)** | 数学 **2 区** | ★★★★ | 中（无"基准测试"栏目） |
| Computers & Operations Research | Elsevier | **4.6**（2024: 4.3） | OR & MANAGEMENT SCIENCE **(Q1, 25/109)** | 工程技术 **2 区** | ★★★（需 OR 化） | 中 |
| European Journal of Operational Research | Elsevier | **7.0**（2024: 6.0） | OR & MANAGEMENT SCIENCE **(Q1, 11/109)** | 管理学 **2 区** | ★★★（需重写骨架） | 中高（官方自述"much more selective"；首轮 20.5 周） |
| Neural Networks | Elsevier | **7.2**（2024: 6.3） | CS, AI **(Q1, 41/210)**；NEUROSCIENCES (Q1) | 计算机科学 **2 区**（Top=是；2024 为 1 区，已下调） | ★★★（需去掉金融框架） | 中 |
| Computers & Mathematics with Applications | Elsevier | 2.5（武大/JCR 2024）vs 2.9（另一镜像，年份不明）→ **冲突，标 UNVERIFIED** | MATHEMATICS, APPLIED (Q1) | 未核 | ★★★★ | 低 |
| Quantitative Finance | Taylor & Francis | 1.9 | BUSINESS, FINANCE Q3；ECONOMICS Q2；MATH, INTERDISC. Q2 | 经济学 **4 区** | 领域最贴但 | **不满足 Q1，排除** |
| Omega | Elsevier | 6.9 | MANAGEMENT (Q1)；OR & MS (Q1) | 管理学 2 区 | ★ | **范围明文排除，排除** |
| Engineering Applications of AI | Elsevier | 9.0 | 四个小类全 Q1 | 计算机科学 **1 区**（Top=是） | ★ | **范围明文排除，排除** |
| Mathematical Finance / Finance and Stochastics / SIAM J. Financial Mathematics | Wiley / Springer / SIAM | 3.6 / 1.10 / 1.40（部分冲突） | 多为 Q1–Q3 | 未核 | ★★ | **理论刊，无定理即拒，排除** |

来源（逐条可点开核对）：

- **JCP 体裁先例**：*"Unveiling the optimization process of physics informed neural networks: How accurate and competitive can PINNs be?"*, **J. Comput. Phys. 523 (2025) 113656** — [ACM DL 条目](https://dl.acm.org/doi/abs/10.1016/j.jcp.2024.113656)（预印本 [ar5iv](https://ar5iv.labs.arxiv.org/html/2405.04230)）。这正是"神经网络求解器的竞争力到底如何"这一体裁，且是负面结论。
- **JCP 范围原文**（要求比较）：*"JCP also encourages the submission of papers that develop innovative methods bridging mathematical, physical modeling and algorithmization… **When addressing problems previously covered by other approaches, a comparison should be provided.**"* — [X-MOL JCP 页](https://www.x-mol.com/8paper/1201710388840566784)（官方指南：[ScienceDirect](https://www.sciencedirect.com/journal/journal-of-computational-physics/publish/guide-for-authors)）
- **JCP 指标与分区**：[武汉大学期刊指南 · JCP](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=13741)（JCR 2024 IF 3.8、Q1/JCP 中科院分区）＋ [jrank JCP](https://jrank.net/journals/j-comput-phys/metrics)（Q1 3/60）＋ [BioxBio JCP](https://www.bioxbio.com/journal/J-COMPUT-PHYS)
- **AMC**：[武汉大学期刊指南 · AMC](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=10020)（JCR 2024 IF 3.4、MATHEMATICS APPLIED Q1、中科院数学 2 区）
- **JSC**：[AbleSci JSC](https://www.ablesci.com/journal/detail?id=rzqVgD)（JIF 2.6、Q1 37/345、**不在 2020/2021/2023/2024 预警名单**）
- **COR**：[AbleSci COR](https://www.ablesci.com/journal/detail?id=DYQQ3D)＋[BioxBio COR](https://www.bioxbio.com/journal/COMPUT-OPER-RES)
- **EJOR**：[AbleSci EJOR](https://www.ablesci.com/journal/detail?id=5JqANp)＋[武大指南 · EJOR](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=12417)＋[jrank EJOR](https://jrank.net/journals/eur-j-oper-res/metrics)
- **Neural Networks**：[AbleSci NN](https://www.ablesci.com/journal/detail?id=pox085)
- **CAMWA**：[武大指南 · CAMWA](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=10046) vs [jrank CAMWA](https://jrank.net/journals/comput-math-appl/metrics)（两者 IF 不一致）
- **Quantitative Finance**（Q3/Q2，中科院 4 区）：[AbleSci QF](https://www.ablesci.com/journal/detail?id=5k9g9r)＋[BioxBio QF](https://www.bioxbio.com/journal/QUANT-FINANC)
- **Omega 范围排除原文**：*"Preference will be given to papers with implications to the practice of management. Submissions of purely theoretical papers are discouraged."* — [AbleSci Omega](https://www.ablesci.com/journal/detail?id=pBlPjD)
- **EAAI 范围排除原文**：*"Submitted papers should report some novel aspects of AI used for a real world engineering application and also validated using some public data sets."* — [AbleSci EAAI](https://www.ablesci.com/journal/detail?id=pP3GOr)
- **SIAM J. Sci. Comput.**（要求"提出方法"）：[X-MOL SISC](https://www.x-mol.com/8paper/1201710651957645312)
- **SIAM JFM**：[jrank SIFIN](https://jrank.net/journals/siam-j-financ-math/metrics)＋[SIAM 官方 flyer](http://siam.org/journals/pdf/flyer_sifin_10.pdf)

> ⚠️ **已知的未核实项（不要当已知用）**：① Clarivate "On Hold"/剔除状态——18 本全部 UNVERIFIED（官方站是 JS 壳）；② APC 价格——全部 UNVERIFIED（出版商 OA 页被 Cloudflare 拦截）；③ 中科院分区有 10 本未核到；④ "该刊是否发表基准测试类论文"——只有 JCP（与 Neurocomputing）核到了具体先例，其余 15 本未核到**不等于**不发表。


## 1.2 为什么主投 JCP

选刊这一步最值钱的信息不是影响因子，而是**"这本刊的投稿指南是否明文要这类稿子、以及它最近有没有发过同类稿子"**。JCP 两项都核到了：

1. **指南明文要求比较**：*"When addressing problems previously covered by other approaches, **a comparison should be provided**. As for any paper in JCP, the efficacy, robustness, computational complexity, as well as reproducibility should be addressed."* —— 这几乎是在描述本文的方法节（等精度比较、FLOP 成本模型、可复现包）。
2. **刚发过同体裁论文**：*"Unveiling the optimization process of physics informed neural networks: How accurate and competitive can PINNs be?"*, **JCP 523 (2025) 113656** —— 一篇"PINN 到底有多准、能不能打"的**负面结论**测量论文。本文的核心形态（用一个可验证的参考解去量神经网络求解器到底行不行）在 JCP 有先例。
3. **JCR Q1 站得稳**：JIF 3.9（JCR 2025），**PHYSICS, MATHEMATICAL 排名 3/60**。另一个小类 CS, INTERDISCIPLINARY APPLICATIONS 是 Q2——投稿时的分类选择有一点策略空间，但 Q1 身份不依赖它。
4. **篇幅对得上**：JCP 建议初投 ≤25 页；本文现在 11 页（会议双栏），扩成期刊版目标 20–24 页，正好在预算内，不必为凑版面注水。
5. **首轮速度可接受**：SciRev 报告首轮约 2.8 个月（样本仅 2 条，参考有限），X-MOL 用户报告平均 6.5 个月——按 **3–6 个月**做计划。

**要做的准备（这正是 JCP 与 AMC 的差别）**：JCP 审稿人不会满足于"我测了很多"，会要一个**可陈述的机制**（P0-B 的命题）和**对高阶重构合法性的讨论**（P0-C 的单调性）。这两件事同时也是把文章从 benchmark 抬到 methods 的东西，不是单纯迎合期刊。

**主要风险（要正视）**：JCP 的学科核心是 computational physics，最优执行/金融 HJB 是边缘应用。这一点无法靠写作消除，只能靠"把贡献写成数值分析结论（重构阶数决定导数/控制精度），把最优执行写成 testbed"来降低——见第四部分的策略建议。

## 1.3 若想 2–4 周内投出去：AMC / JSC

如果时间预算不允许做命题与非多项式算例，那么结论会变，目标也该变：

- **Applied Mathematics and Computation**：JCR Q1（MATHEMATICS, APPLIED），IF 3.4（JCR 2024，已从武大图书馆指南核实；2025 数值未核到）。它是"数值格式 + 大量数值实验 + 与既有方法比较"的常规发表地，**本文当前形态几乎不需要改写**——这是它最大的价值。
- **Journal of Scientific Computing**：JCR Q1（应用数学 37/345），JIF 2.6（JCR 2025），**不在任何一轮中科院预警名单**（2020/2021/2023/2024），用户报告 4–8 周。风险最低的一档。
- **Computers & Operations Research**：JCR Q1（OR & MS 25/109），JIF 4.6，报告周期约 3.3 个月——**OR 一档里最快的**，代价是按"决策支持/算法选择"改写。

**要接受的代价**：AMC/JSC 的中科院分区都是**数学 2 区**；JCP 是**物理与天体物理 2 区**（小类 1 区）。

## 1.4 备选、以及已被证据排除的选项

**可以投但要改写骨架**：

- **EJOR**（JIF 7.0，OR & MS 11/109）：官方自述 *"EJOR is much more selective than in the past due to the growing inflow of submissions and limited volume"*，首轮 20.5 周。要投它，需把"哪个求解器更快"重述为"给定精度目标与市场参数，决策者如何选方法"，把 `tab_guide` 做成可执行规则 + 成本—精度前沿分析。**2–4 周的改写，不是换模板。**
- **Neural Networks**（JIF 7.2，CS-AI 41/210；中科院计算机 2 区但 Top=是）：可以投，但**必须去掉金融框架**，改成"神经求解器的可微性/控制精度"这一学习系统命题（OOD 与重构阶数结果都能挂上去）。注意中科院 2024 是 1 区、2025 下调为 2 区。

**已被证据排除（不要再花时间）**：

| 期刊 | 排除依据 |
|---|---|
| **Omega** | 范围原文：*"Preference will be given to papers with implications to the practice of management. Submissions of purely theoretical papers are discouraged."* —— 数值精度研究与"管理实践含义"无关 |
| **Engineering Applications of AI**（唯一的中科院计算机 1 区候选） | 范围原文：*"Submitted papers should report some **novel aspects of AI** used for a **real world engineering application**…"* —— 本文不提新 AI、不是工程应用、还是负面结果，三条全不满足 |
| **SIAM J. Scientific Computing** | 范围要求 *"computational results demonstrating the effectiveness of **the proposed techniques**"* —— 本文没有"proposed technique"（除非把三次重构写成一个方法，见第四部分策略） |
| **SIAM J. Financial Mathematics / Mathematical Finance / Finance and Stochastics** | 理论刊；无定理的数值基准是典型 desk reject（后两本的判断为**推断**：出版商页面被拦截，未取到原文） |
| **Quantitative Finance** | JIF 1.9，**BUSINESS, FINANCE 仅 Q3**（SSCI），中科院**经济学 4 区** —— 既非 Q1，分区也低 |
| **Journal of Computational Finance** | 选题最贴，但 JIF 仅 0.5，且 **SCIE/ESCI 归属未核实** |
| **J. of Machine Learning for Modeling and Computing** | 选题极贴，但**仅 ESCI 收录、中科院分区未分配** —— 国内多数单位不计 |

## 1.5 「JCR 1 区」与「中科院 1 区」是两件事

你写的是 JCR 1 区，我按 JCR 四分位处理。但国内多数单位考核看**中科院分区**，两套体系在本文候选期刊上结论不同（中科院为 **2025 版**，2025-03-20 发布）：

| 期刊 | JCR 分区 | 中科院 2025 大类 | Top | 预警名单 |
|---|---|---|---|---|
| Journal of Computational Physics | Q1（物理:数学物理 3/60） | 物理与天体物理 **2 区**（小类 物理:数学物理 **1 区**） | 是 | 未核 |
| Applied Mathematics and Computation | Q1（应用数学） | 数学 **2 区** | — | 未核 |
| Journal of Scientific Computing | Q1（应用数学 37/345） | 数学 **2 区** | 否 | **不在**（2020/21/23/24） |
| Computers & Operations Research | Q1（OR & MS 25/109） | 工程技术 **2 区** | 否 | **不在** |
| European Journal of Operational Research | Q1（OR & MS 11/109） | 管理学 **2 区** | 否 | **不在** |
| Neural Networks | Q1（CS-AI 41/210） | 计算机科学 **2 区** | 是 | **不在** |
| Engineering Applications of AI | Q1（四个小类） | 计算机科学 **1 区** | 是 | **不在** |
| Quantitative Finance | Q2/Q3 | 经济学 **4 区** | 否 | **不在** |

**含义**：若硬指标是"**中科院大类 1 区**"，这份候选名单里**只有 EAAI 满足——而它的范围声明明文排除了本文**。也就是说：

> **没有一本期刊同时满足"JCR Q1 + 中科院 1 区 + 体裁契合"。** 要么接受中科院 2 区（JCP/AMC/JSC/EJOR/COR），要么改变文章的自我定位（例如把贡献重述为 AI 方法学，冲计算机 1 区），要么重新审视"1 区"的口径。

另需注意：两套分区每年都变（Neural Networks 从 2024 的 1 区降到 2025 的 2 区，EAAI 反向升到 1 区）。**投稿前请以当年官方分区表复核一次。**

## 1.6 如果先投 CISP-BMEI 再扩展投期刊

会议（IEEE Xplore 检索）之后再投期刊是常规路径，但有两条边界：

1. **必须有实质增量**。会议版与期刊版的重合度通常被要求低于 ~70%（各刊规定不同），且期刊版**必须引用会议版**。本文的会议版已经 11 页，若期刊版只是加几段，会被判重复发表。
2. **本文件里的 P0 条目本身就是增量**：口径重算、命题、非多项式算例、LSMC/高层次概率基线、编译版计时——这些足以构成"70% 以上新内容"，而且是真正提升文章层级的内容。

---

# 第二部分 · 这篇文章现在的成色

## 2.1 已经达到 Q1 水准、不要动的部分

这些是这篇文章真正的资产，改写时**不要为了"看起来更学术"而削弱它们**：

1. **精确参考解 + 独立校验**。Riccati 闭式解与三个独立闭式（有限 Q_T 闭式、Q_T→∞ 的 coth 极限、Almgren–Chriss 轨迹）互校，误差 1e-12 ~ 1e-9 量级（`src/ac_problem.py`）。
2. **每个数字都可溯源（但"可溯源"≠"正确"）**。`verify_paper_numbers.py` 7/7 断言通过、`check_table_drift.py` 12 张表逐位一致、`results/` 55 个结果文件（`results/journal_audit.json`）。⚠️ 这套校验保证的是"稿件与 `results/` 一致"，**不保证 `results/` 与物理事实一致**——P0-A2 正是从这个缝隙里漏出去的。这个区分在写期刊版时必须显式化。
3. **诚实的负面结果与限定**。五条 limitations 写在正文里，包括对自身最不利的那条（网格计时用的是 NumPy、其实现率还在随 N 上升）。
4. **明确的"我们没做什么"**。LSMC 未跑、只用一个架构族、真实数据只做管道刻画——都写明而非回避。
5. **发布包干净**。仓库 `ridleyton/HJB-Solver-Comparison` 已公开（105 文件），凭据/绝对路径扫描 0 命中（`results/scan_remote.json`、`results/check_release.json`）。

## 2.2 体裁判断（这决定选刊）

这篇文章的真正贡献是**一条可操作的方法学结论**（"控制精度由重构阶数决定，不由求解器族决定"）+ **一套受控测量**。它不是：
- 新算法（论文自己声明不主张任何新方法）；
- 新理论（目前没有命题）；
- 新应用（金融部分只到系数标定与管道缺陷）。

这个体裁在 Q1 里能活，但**必须靠"测量质量 + 可操作结论 + 可复现"三件事立住**，而这恰恰要求上面的 P0-A（口径）不能有瑕疵——因为一旦数字被质疑，剩下的就只有"一份很用心的基准测试"。

---

# 第三部分 · 差距清单

## 3.0 P0-A：误差口径不统一——你的头条数字有一半是"平方误差"口径

> **✅ 已于 2026-10-14 修复（阶段 0）。** 修复方式与前后数字对照见 [`STATUS.md`](STATUS.md) 的 "Stage 0" 一节；本节的诊断记录保留作为判断依据。现在口径定义在 [`src/metrics.py`](src/metrics.py)，重算脚本 [`src/recompute_metrics.py`](src/recompute_metrics.py)，回归守卫 [`src/check_order_convention.py`](src/check_order_convention.py)；正文断言由 7 条扩到 **25 条并全部通过**。

**这是我这轮复核里发现的最重要的一处问题，也是"改起来最便宜、不改最贵"的一处：所有数字都在，重算不需要重跑任何实验。**

### 证据

`exp_control_extraction.py`（生成 `tab_extract`）与 `exp_control_order.py`（生成 `tab_control`）拟合的是**误差平方**的对数斜率：

```python
# src/exp_control_extraction.py
err = float(np.mean((g - exact) ** 2) / scale_f)   # ← 这是 MSE，不是误差
# src/exp_control_order.py
"value_err": v_mse / v_scale, "control_err": u_mse / u_scale   # ← 同样是 MSE
```

而 `exp_reconstruction_order.py`（生成 `tab_order`）拟合的是**误差范数**：

```python
# src/exp_reconstruction_order.py
return float(np.sqrt(np.mean(err**2)) / scale)     # ← RMS，真正的误差
```

由于 log(E²) = 2·log(E)，**同一个物理量的斜率会差 2 倍**。我直接从 `results/*.json` 重新拟合验证（脚本见附录 A）：

| 量 | 论文现在的数字 | 口径 | 换成误差范数口径 |
|---|---|---|---|
| 控制误差收敛阶 | "approximately 2.00" | MSE 斜率 | **1.00**（与 `tab_order` 的 degree-1 = 1.03–1.10 一致） |
| 多线性重构 | "second order on every function tested" | MSE 斜率 | **一阶** |
| 三次重构 | "fourth order"（4.4–5.6） | MSE 斜率 | **2.2–2.8**；而 `tab_order` 的 degree-3 报 3.29–3.53（探针步长不同，见下） |
| "four to five orders of magnitude more accurate" | 摘要 | MSE 比 | **1.4×10²–4.5×10²（约 2.1–2.6 个数量级）** |
| 控制/价值误差比 | "184"，摘要"10 to 684" | MSE 比 | **13.6**；**3.2–26** |
| 三次重构后的比值 | 2.8 | MSE 比 | **1.7** |

**复核数据（`results/exp_control_extraction.json`，固定网格上的 fd/spline 比）**：

```
h=0.0333  quartic: MSE 比 3.6e+04 → RMS 比 1.9e+02
h=0.0333  exp.   : MSE 比 2.1e+05 → RMS 比 4.5e+02
h=0.0333  trig.  : MSE 比 2.0e+04 → RMS 比 1.4e+02
```

### 还有一处"表比表"不一致

`tab_extract` 的 spline 用探针步长 **eps = 0.25h**，`tab_order` 用 **eps = 1e-4·h**（几乎等于对插值函数求精确导数）。两者测的不是同一个算子，因此三次重构给出 2.2–2.8 与 3.3–3.5 两个不同的阶。正文同时引用这两张表却未说明差异来源。

### 顺带一处残留

正文两处把比值写成 **184**（`main.tex:490`、`main.tex:851`），但决策表 `tab_guide.tex:25` 仍是 **185**。这是因为 `tab_guide` 是手写表，`check_table_drift.py` 只覆盖自动生成的表，`verify_paper_numbers.py` 只断言正文——两套校验的缝隙正好漏掉它。

### 为什么 Q1 审稿人一定会抓

数值类期刊的审稿人第一件事是问"你的 error 是哪个范数"。一旦发现阶数取自平方误差，"second order"和"four to five orders of magnitude"两句话同时失效——**而这两句正好是摘要的核心卖点**。

### 怎么补（1–2 人日）

1. 在方法节加一段**度量定义**：明确"所有收敛阶为 log‖e‖ 对 log h 的最小二乘斜率"，并给出误差范数（建议相对 L² 范数 ‖·‖₂/‖·‖₂^ref）；R² 降级为辅助指标。
2. 从 `results/*.json` 重算全部阶与比值（数据都在，不需重跑实验）。重算后**主结论方向不变**：控制误差确实远大于价值误差、确实由重构阶数决定、三次重构确实显著更好——只是数量级要改。
3. 把 `tab_extract` 与 `tab_order` 的探针步长统一，或明确写出两者差异并给出量化解释。
4. 修 `tab_guide` 的 185 → 与正文一致的数，并把手写表纳入 `check_table_drift` 的范围（或至少加一条断言）。
5. 加一条自动化断言防回归：把本次的复核脚本纳入 `verify_all.py`。

---

## 3.0b P0-A2：`tab_bsde` 的 p=4 行是 d=8 的数字（自动生成器的取行 bug）

> **✅ 已于 2026-10-14 修复（阶段 0）。** `load_run` 现在按维度过滤，并新增按维度断言的校验（`verify_paper_numbers.py` 会同时核对 d=4 的 3.3% 与 d=8 的 0.3%）。表已重生成：p=4 行为 0.033 / 5.21e+00 / 192 s。

**这是我这轮发现的第二处、也是更硬的一处问题：一张已经"机械校验通过"的表里有一个错数字。**

### 证据链

1. `results/exp_bsde_dims.json` 里有**两条**记录：`d=4` 与 `d=8`（两次运行写进了同一个文件）。
2. `src/make_bsde_table.py:115-142` 的 `load_run(p, fname, n_steps)` **只在文件内选 `v_err` 最小的那一行，没有按维度过滤**：

```python
for r in rows:                      # ← rows 含 d=4 和 d=8 两条
    ...
    if best is None or cand["v_err"] < best["v_err"]:   # ← 谁小选谁
        best = cand
```

   于是 `p=4` 请求拿到的是 `d=8` 的记录（d=8 的误差本来就小一个数量级）。

3. 实测两条记录的真实值：`d=4`：best v_err **0.03274**（epoch 2000，floor 2.012e-04，0.0961 s/epoch）；`d=8`：best v_err **0.003478**（epoch 6000，floor 4.023e-04，0.1159 s/epoch）。

### 影响（全部集中在 `tab_bsde.tex` 的 p=4 行）

| 列 | 现在表里显示 | 正确值 |
|---|---|---|
| rel. err at x₀ | **0.003**（0.3%） | **0.033**（3.3%） |
| resid. at stop | **7.17e-01** | **5.21e+00** |
| train (s) | **696** | **≈192**（0.0961 × 2000 epoch） |

**关键点：正文是对的，表是错的。** 正文写 "At $p=1$ and $p=4$ its best value error is $5.3\%$ and $3.3\%$"（`main.tex:671`），与 `analysis_bsde_dims.md` 一致；残差倍数 2.6×10⁴ 也是按正确的 d=4 floor 算的。所以现在**同一篇文章的正文与表格互相矛盾**，且矛盾方向**削弱了本文自己的结论**（表里把 deep BSDE 写得比实际好 10 倍）。修正后结论更强。

### 为什么三套校验都没抓到

| 校验 | 为什么漏过 |
|---|---|
| `check_table_drift.py` | 它"从 `results/` 重新生成表再比对"——生成器有 bug，重新生成还是同一个错数字，**自洽但不正确** |
| `verify_paper_numbers.py` | 它断言的是**正文**，正文本来就是对的那一份 |
| `check_release.py` | 只查凭据与路径 |

**这是一个方法论教训，值得写进项目的 STATUS**：*"可复现"不等于"正确"——从错的上游重新生成，会稳定地复现同一个错误。*

### 怎么补（约 0.5 人日）

1. 在 `load_run` 里按维度过滤：`if r.get("d") is not None and int(r["d"]) != p: continue`；并加一条断言"选中行的 d 必须等于请求的 p"，让同类错误以后直接抛错而不是静默取错行。
2. 重新生成 `tab_bsde.tex` 并重编译（顺带核对 caption 里 "0.11–0.15 s per epoch" 的区间是否要含 d=4 的 0.0961）。
3. 给 `check_table_drift.py` 加一条**跨表一致性**检查：正文提到的比例（如 3.3%、2.6×10⁴）必须能由表内数字复算出来。

---

## 3.1 P0-B：头部主张没有理论结果

**现状**：全文 0 个定理/命题/引理。核心主张"控制精度由重构阶数决定，与价值精度无关"完全靠数值观察支撑（`tab_extract`、`tab_order`、`tab_control`）。

**Q1 审稿人会怎么说**：*"The central claim is a numerical observation. A proposition establishing the order of the recovered gradient as a function of the reconstruction degree, and the resulting control/value error ratio, would make this a methods contribution rather than a benchmarking exercise."*

**怎么补（1–2 人周，纯推导，不需算力）**：

- **命题 1（梯度恢复阶）**：给定光滑 V、节点值误差 O(h^{k+1})、k 次张量积重构，则恢复的梯度满足 ‖∇R_h V − ∇V‖ = O(h^k)；并说明在**求值点位置**上的差别（节点/单元中心 vs 一般点）——你现在的实验恰恰是在**随机点**上测的，这是比"只在节点上成立"更强的结论，值得单独陈述。
- **命题 2（控制→轨迹→目标值传播）**：由 δu 到库存路径偏差到目标损失 O(‖δu‖²) 的传播率。
  ⚠️ **注意**：正文现在写"the fitted orders are what the envelope theorem predicts rather than merely consistent with it"，但实测的（同口径下）控制误差阶约 1.0、路径误差阶约 1.8–2.0、目标损失阶约 2.1（我在 `results/exp_control_consequence.json` 上重拟合得到）。**"包络定理预测"这句话目前是断言，不是推论**，而且数字之间还差着一阶。要么补出推导，要么把这句改弱——留着它比删掉它危险。
- **推论（可操作建议的定理化）**：控制精度可通过提高重构阶数获得，而网格空间加密的收益受价值误差阶限制——这正是论文给从业者的建议，把它写成推论，文章立刻从"测量报告"变成"方法学论文"。

**文献定位（必做，见 P1-2）**：梯度恢复的高阶性在数值分析里是经典结果（ENO/WENO、超收敛与梯度恢复）。**必须主动引用并说清增量在哪**——增量不是"高阶插值能提高梯度精度"（这是教科书），而是①在 HJB/最优执行语境下把"控制精度"与"价值精度"解耦并量化，②给出执行层面的后果（持仓偏差、目标损失），③给出"先改重构、再改网格"的可操作次序。

## 3.2 P0-C：单调性与粘性解框架缺失

**现状**：论文推荐"把重构换成三次"，但**高阶插值不是单调格式**。全文未出现 Barles–Souganidis、单调性、粘性解等概念。

**为什么重要**：HJB 的收敛性理论建立在"单调 + 一致 + 稳定"的格式上。审稿人会问：
- 高阶重构是否破坏价值格式的单调性？
- 在**控制进入算子**的情形（策略迭代、一般 HJB），恢复出的控制会反馈进 PDE，此时单调性怎么办？
- 终值附近控制发散（clearing 问题最后一刻 |u|→∞），高阶插值会不会产生振荡/过冲，让"更准"变成"不可用"？

**本文其实有一个很好的答案，但没写出来**：在 LQ benchmark 里 Hamiltonian 的最大值点有解析解，重构只用于**事后提取控制**，不进入价值方程——所以价值格式的单调性不受影响。这个论证必须写出来，并说明它的**适用边界**（一般 HJB 不成立）。

**怎么补（3–5 人日）**：
1. 增加一小节"重构何时是后处理、何时进入算子"，给出两种情形的区分与后果。
2. 加一个**保单调/限幅的重构变体**（WENO 型或简单 limiter），测它在同一批测试函数上的阶与单调性代价——这会成为论文里第二个"可操作"的结论。
3. 在终值附近做一次专项检查：控制的最大偏差、是否出现符号翻转、限幅前后对比。

## 3.3 P0-D：基准问题的解是二次多项式，网格法被天然优待

**现状**：V 是二次型，Riccati 给出精确解；`tab_grid` 显示达到 R² ≥ 0.9999 所需分辨率 **N\* = 21 在所有维度都一样**。

**这意味着什么**：二次函数被二阶格式近乎精确地表示，所以"网格法在等精度下更快"是在一个**对自己最有利的基准**上得到的。审稿人会说：*"The test problem's solution is a quadratic polynomial; the spatial resolution requirement is therefore dimension-independent and the comparison is biased toward the grid method."*

**怎么补（1–2 人周）**：
1. 至少一个**非多项式解**的求解器对比（不是只测重构阶数，而是跑完整对比）。可选：非线性临时冲击（Guéant 型，Hamiltonian 非线性）、随机波动率/双因子、或带边界层的构造解（manufactured solution）。
2. 报告 N\* 在该算例下**如何随维度变化**——如果 N\* 开始随 d 上升，"N\*=21 与维度无关"就从结论变成"这个基准的性质"，这本身就是一条有价值的发现。
3. 现有 Merton 算例（非多项式、闭式幂函数）目前只用于重构阶数，把它升级成完整的求解器对比是**成本最低的路径**。

## 3.4 P0-E：缺关键基线

**现状**：网格（显式 FD、Howard 策略迭代、隐式 LOD）、PINN、PI-PINN、deep BSDE。LSMC 在 Related Work 里明确写了"未跑"。

**问题**：文章的核心边界结论是"d ≥ 8 时神经网络是唯一能跑的方法"。但这句话在有 LSMC、deep splitting、multilevel Picard 的世界里不成立——**审稿人会立刻举出反例**。

**怎么补（2–3 人周）**：
1. **LSMC（Longstaff–Schwartz）**：最自然的缺席者，无需网格、可到高维，且是"经典方法"。在 p=1..8 上与 PINN/deep BSDE 同表对比，能直接把"唯一能跑"改写成"在哪些维度、哪种方法更划算"——**这是比现状强得多的结论**。
2. **至少一个现代高层次概率方法**：deep splitting（Beck 等，已引用未跑）或 multilevel Picard（E–Hutzenthaler–Jentzen；你已引用 Hutzenthaler 2022）。二者都是"有复杂度理论"的方法，正好回答"高维到底该用什么"。
3. 保持 deep BSDE 现在的诚实定位（用其参考版本而非最强变体）——这个说明要保留。

## 3.5 P0-F：deep BSDE 报的是 oracle 选点

**证据**：`tab_bsde` 说明"the value is quoted **at its best point during training**"，而"best"是相对精确解判定的。`analysis_bsde_steps.md` 里已经写明：残差阈值法不需要参考解（≈1.76e-3），但该阈值**跨维度不可移植**（最优比例 26% / 3.9% / 0.8%，相差 31 倍）。

**审稿人会怎么说**：*"The reported 0.35–5.3% error is obtained by selecting the epoch that minimises the error against the exact solution. A practitioner cannot do this."*

**怎么补（2–4 人日）**：
1. 定义一个**不需要参考解**的停止准则（保留独立路径批次的验证估计，或用标定过的高偏/低偏估计监控 V(x₀,0)）。
2. 同时报三个数：① 该准则下的价值误差、② oracle 最优点的误差、③ 两者之差（即"停止准则的代价"）。这比现在只报 oracle 数**更有说服力**，因为差额本身就是一条结果。
3. 补 2–3 个种子（现在每个维度一次运行、一个初始状态）。

## 3.6 P1-1：统计强度与表间一致性

- PINN 结果只有 **3 个种子**，且 R² 是重尾统计量：p=4 是 **−57.02 ± 46.58**，p=5 是 **−615.78 ± 0.68**。用均值±标准差描述重尾量会误导，**建议改报中位数与四分位距**，并把 R² 之外再报相对 L² 误差。
- **表间不一致**：`tab_main` 的 p=4 基线架构是 **−57.02**（3 种子均值），`tab_arch` 同一配置是 **−3.28**（单次运行）。这两处必须在对齐口径（同种子集/同评测点）后重报，否则审稿人会当成矛盾。
- 建议：种子数提到 5–10，固定调参协议（预算对齐、学习率/采样固定），并给出"同算力预算下"的对比表。

## 3.7 P1-2：文献定位（40 条 → 目标 55–70 条）

现有 40 条覆盖了 PINN/deep BSDE/算子学习/最优执行，但**缺三块最要紧的**：

| 缺口 | 必须补的代表性文献 | 为什么 |
|---|---|---|
| 高阶重构与梯度恢复 | Harten 等 (ENO, 1987)；Shu & Osher (1988/1989)；Liu–Osher–Chan (WENO, 1994)；Zienkiewicz & Zhu (超收敛斑块恢复, 1992)；DG 后处理（Cockburn–Luskin–Shu–Süli, 2003） | 否则"提高重构阶数可以提高梯度精度"会被读成新发现，实际是经典结果——**主动引用反而抬高本文的增量**（HJB 语境 + 控制精度解耦） |
| HJB 数值分析 | Barles & Souganidis (1991)；Crandall & Lions (1983)；Falcone & Ferretti（半拉格朗日）；Bungartz & Griebel（稀疏网格, 2004） | 单调性/收敛性讨论的引用基础（见 P0-C） |
| 最优控制的误差估计 | Falk (1973)；Hinze 等《Optimization with PDE Constraints》；Malanowski/Rösch 的控制误差估计 | 控制误差与价值误差的**阶差**在 PDE 约束优化里有成熟理论，必须定位 |
| 高层次概率方法 | E–Hutzenthaler–Jentzen（multilevel Picard）；Beck 等（deep splitting，已引用）；Huré–Pham–Warin (2020)；Germain–Pham–Warin（deep BSDE 误差分析） | 支撑 P0-E，并给出"高维该用什么"的完整图景 |

**另加一段**：显式写出"哪些是经典结果、哪些是本文新增"。这一段写好了是加分项，写不出来就是拒稿理由。

## 3.8 P1-3：边界与截断域未讨论

**现状**：状态域取 `[-1,1]^d`（`src/ac_problem.py:241`），但全文**没有一句**讨论截断域大小的影响、边界条件的处理、以及边界单元对误差统计的贡献。

**为什么会被问**：评测点是域内均匀随机采样，因此落在**边界单元**内的比例随维度上升：N=21 时每个轴有 20 个单元、其中 2 个贴边，d=5 时落在至少一个边界单元内的概率是 1−(18/20)⁵ ≈ **41%**。也就是说，高维的控制误差统计有相当一部分来自边界单元——那里恰恰是多线性/三次重构最弱、最需要单侧处理的地方。

**怎么补（3–5 人日）**：分开报"内部点/边界单元"的误差与阶；说明边界处理方式；做一次域大小敏感性（域放大/缩小对结论的影响）；说明初始状态 x₀ 恰在域角上这一事实是否影响 deep BSDE 的对比。

## 3.9 P1-4：成本比较的硬件公平性

**现状**：网格用 NumPy（CPU）、神经网络用 PyTorch（GPU）。论文用 FLOP 成本模型正面回应（这是对的做法），但也自认"网格实现率还在随 N 上升、编译内核会更快"。

**Q1 审稿人会说**：*"A 5–50× speed claim based on a NumPy implementation against a GPU-trained network is not a fair comparison."*（这句话对你的结论方向其实**有利**——编译后会更快——但不说清楚就是软肋。）

**怎么补（约 1 人周）**：
1. 用 **JAX 重写网格求解器**（与 PINN 同框架、同设备），给出 GPU/CPU 两套计时；现在的 FLOP 模型保留作为交叉校验。
2. 报**达到的 GFLOP/s 与峰值效率**，把"NumPy 的实现率"从脚注升级成一张表。
3. 若时间不允许，至少把"编译版网格会更快，因此本文的速度结论是保守的"从限制条款升级为**一句带量级估算的论证**。

## 3.10 P1-5：真实数据的部分偏薄

**现状**：9 只美股、两个月的分钟线、14 天测试窗；只做系数标定（σ、η=半 Roll 价差）、CFL 步数放大（23–78×）、OOD 现象与缺陷消融；明确不宣称任何盈利能力。

**判断**：如果投数值期刊，这部分是"适用范围分析"，完全够用（而且诚实是加分项）。如果投金融期刊，则太薄。

**建议的一条高价值扩展（1–2 人周）**：在真实标定系数下，把"多线性 vs 三次重构"落到**执行层面**：用真实分钟数据模拟两条控制路径，比较实现差额（implementation shortfall）/滑点，把一个抽象误差变成**货币单位**的差异。这一步不需要新的理论，却能把文章的说服力提高一个档次，而且你已经有两块现成的拼图（`exp_control_consequence.py` 的执行路径框架 + `exp_real_calibration.py` 的真实系数）。

## 3.11 P1-6：可复现性发布（差一步）

- 仓库已公开 ✅、`.gitignore` 已建 ✅、49 项检查通过 ✅。
- 缺：**归档 DOI**（Zenodo/OSF 打 tag 发布，期刊普遍要求可引用的永久标识）、**版本号与论文版本对应**、**表→脚本→结果文件的三列对照表**（现在靠 `check_table_drift.py` 机械保证，但读者看不到这条路径）。
- 市场数据不能随包分发已说明 ✅，建议再补一句"派生量（半 Roll 价差、波动率）已随仓库发布，因此标定表可在无原始数据时复核"。

## 3.12 P2：呈现与格式

| 项 | 现状 | 期刊需要 |
|---|---|---|
| 模板 | IEEEtran conference | 目标期刊模板（Elsevier `elsarticle`） |
| Highlights | 无 | Elsevier 多数期刊要求 3–5 条（每条 ≤85 字符） |
| CRediT / 利益冲突 / 数据可用性 | 只有 Reproducibility 段落 | 三项独立声明 |
| 图 | 6 张 | 建议 8–12 张：误差—工作量 Pareto 图（含等精度线）、决策树图（把 `tab_guide` 图形化）、收敛图（带拟合斜率与置信带）、内部/边界误差分离图、执行后果图 |
| 表 | 7 张 | 建议把缺陷目录（`tab_ablation`）移入补充材料，正文留摘要 |
| 符号表 | 无 | 建议加 |
| 篇幅 | 11 页双栏 | JCP 建议初投 **≤25 页**（已核实）；AMC 通常 20–30 页量级。需要把 `archive/paper_long_v21.tex` 的长版材料与新结果合并，不是简单加几段 |
| 语言 | — | 母语润色（这是唯一一项你原本就列为"待办"的事） |

---

# 第四部分 · 改进路线图

按"先修可信度、再补贡献、最后包装"排序。工作量按**净投入**估算。

### 阶段 0（1–2 周）· 止血：把口径和一致性修好

| 任务 | 产出 | 对应条目 |
|---|---|---|
| 统一误差口径，重算全部阶与比值 | 修订后的 tab_extract / tab_control / 摘要数字 | P0-A |
| 统一/解释两处探针步长差异 | 方法节一段 + 表注 | P0-A |
| **修 `make_bsde_table.py` 的取行 bug 并重新生成表** | 修正后的 tab_bsde（p=4 一行三处数字） | **P0-A2** |
| 修 `tab_guide` 的 185；把手写表纳入漂移检查 | 校验脚本升级 | P0-A |
| 对齐 `tab_main` 与 `tab_arch` 的种子/评级口径 | 重报的 p=4 数字 | P1-1 |
| deep BSDE 的无参考解停止准则 + 三档报数 | 新版 tab_bsde | P0-F |

**这一阶段做完，文章的"可信度"就达标了**；即使暂时不投，也不会被一算就穿。

### 阶段 1（3–5 周）· 补贡献：从 benchmark 升级为 methods

| 任务 | 产出 | 对应条目 |
|---|---|---|
| 命题 1（梯度恢复阶）+ 命题 2（传播率，或改写现有断言） | 新增理论节（半页） | P0-B |
| 单调性/重构角色讨论 + 保单调变体的数值对比 | 新增小节 + 1 图 1 表 | P0-C |
| 非多项式解的完整求解器对比（优先 Merton 升级版） | 新增算例 + N\*(d) 曲线 | P0-D |
| LSMC + 一个现代高层次概率方法 | 扩表 + 改写"唯一能跑"的表述 | P0-E |
| 边界/截断域分析 | 1 表 | P1-3 |
| 文献补齐与"经典 vs 新增"段落 | 参考文献 55–70 条 | P1-2 |

### 阶段 2（2–3 周）· 包装与投稿

| 任务 | 产出 |
|---|---|
| JAX 网格求解器 + 硬件公平计时表 | 新版成本节（若要冲 JCP 则必做） |
| 真实数据上的执行差额实验（可选但强烈建议） | 新的执行后果小节 |
| 期刊模板 + Highlights + CRediT + 数据声明 + 符号表 + 图增补 | 投稿包 |
| Zenodo 归档 + 表→脚本映射表 | 可引用 DOI |
| 母语润色 | 终稿 |

**投 JCP（主投）**：三个阶段都要做，阶段 2 的硬件公平性与篇幅控制（≤25 页）是硬要求。
**投 AMC / JSC（快投）**：阶段 0 + 1 基本足够（阶段 2 只做包装）。

### 一条贯穿三个阶段的写作策略

本文最有普适性的主张是**数值分析层面的**：*重构算子的阶数决定导数/控制精度；时间加密救不了它；网格成本是 N^p 且受 CFL 约束，而按真实价差标定会让步数再涨 23–78 倍。* 金融是最优执行这个**试验台**，不是卖点本身。

因此期刊版的组织应当反过来：**先立命题与机制，再让 LQ 最优执行为它提供可验证的实例**。现在的写法（"哪个求解器赢"）在金融与 OR 期刊会招来"缺乏新意"的 desk reject，而在数值期刊又显得贡献不足——同一份内容，**换个主次关系，两个问题一起解决**。


---

# 第五部分 · 需要你确认的事

1. **口径**：硬指标到底是 **JCR Q1** 还是**中科院 1 区**？二者在本文候选期刊上结论不同（见 §1.5）。这决定是否需要换赛道。
2. **会议与期刊的关系**：还投不投 CISP-BMEI？若已投/将投，期刊版必须按 §1.6 处理增量与引用。
3. **是否愿意做理论部分**：P0-B/P0-C 是这篇文章能否进 JCP 的分水岭。若不做，AMC 是上限（这不是坏结果）。
4. **真实数据的使用权限**：分钟数据不能随包发布，但**执行差额实验**需要它。确认数据在研究环境里可用即可，不需要发布。
5. **时间预算**：上面的路线图是 6–10 周量级。如果只有 2 周，那就只做阶段 0 + 投 AMC/JSC——**这也是一个诚实且合理的选择**（JCR Q1 身份不变，只是中科院同为 2 区、影响力低一档）。

---

# 附录 A · 本文件断言的复核命令

```powershell
cd "rework/"

# 1. 误差口径：从 results/*.json 重新拟合，比较 MSE 口径与 RMS 口径
python src\check_order_convention.py

# 2. deep BSDE 取行 bug：exp_bsde_dims.json 里有 d=4 与 d=8 两条记录，
#    而 load_run 不按维度过滤、只取 v_err 最小的一条 —— 直接看这个文件即可确认
python -c "import json,pathlib; d=json.loads(pathlib.Path('results/exp_bsde_dims.json').read_text()); print([(r['d'], r['exact_floor'], r['best_V_rel_err']) for r in d['rows']])"

# 3. 论文自身的校验（现状：均为通过——注意它们查不出 P0-A 与 P0-A2）
python src\check_tex.py                 # 结构：0 问题
python src\verify_paper_numbers.py      # 正文断言：7/7
python src\check_table_drift.py         # 表格与 results/ 逐位一致（上游错则一起错）
python src\journal_audit.py             # 0 blocker / 2 manual / 1 enhance（results/journal_audit.json）
```

**口径复核的核心数据**（`check_order_convention.py` 输出，已在本机跑过）：

```
A. exp_control_extraction.json（生成 tab_extract）
   multilinear(fd)   slope(log E)=2.09~2.31   slope(log sqrt(E))=1.04~1.16
   cubic(spline)     slope(log E)=4.20~4.75   slope(log sqrt(E))=2.10~2.37
B. exp_reconstruction_order.json（生成 tab_order）
   degree 1  报 1.03~1.10    degree 3  报 3.29~3.53    degree 5  报 5.08~6.34
```

→ **A 与 B 报的是同一个物理量，但 A 是平方口径、B 是范数口径**；这就是 P0-A。

## 附录 B · 本轮新增文件

| 文件 | 说明 |
|---|---|
| `rework/JOURNAL_Q1_TARGET_AND_GAPS.md` | 本文件 |
| `rework/src/check_order_convention.py` | 误差口径复核脚本（只读 `results/*.json`，不改动任何实验） |
| `rework/results/_order_convention_check.py` | 首次核对时写下的副本，会话沙箱不允许删除，已改为指向上面那个脚本的说明文件（无作用） |

**本轮未改动任何原有文件**：`paper/main.tex`、`paper/tables/*`、`src/*`、`results/*.json` 均保持原样。上面所有问题都是**读**出来的，修复方案是建议、尚未实施。

---

## 附录 C · 本次核对直接发现的两处硬问题（一页速览）

| # | 位置 | 现在 | 应为 | 性质 |
|---|---|---|---|---|
| 1 | `tab_extract` / `tab_control` / 摘要 / 结论 | "order 2.00"、"second order"、"four to five orders of magnitude"、"184"、"10 to 684" | 同口径下：order ≈ 1.0、一阶、约 2.1–2.6 个数量级、"13.6"、"3.2–26" | **口径**：拟合的是误差平方，而 `tab_order` 拟合的是误差范数 |
| 2 | `tab_bsde` 的 p=4 行 | rel err 0.003、resid 7.17e-01、train 696 s | 0.033、5.21e+00、≈192 s | **取行 bug**：`load_run` 不按维度过滤，选中 d=8 的记录 |

两处的共同点：**论文自身的三套校验全部通过**。第 1 处是因为口径写错但两边一致；第 2 处是因为生成器有 bug、"重新生成"只会复现同一个错误。

对投稿的实际含义：
- 第 1 处若被审稿人发现，损伤的是**整篇文章的可信度**（头条数字口径不实）；
- 第 2 处若被审稿人发现，损伤的是**这篇论文的结论强度**（修正后 deep BSDE 在 p=4 的误差从 0.3% 变成 3.3%，"deep BSDE 打不过网格"这句话反而更硬）。

**先做这两件事，再谈选刊。**
