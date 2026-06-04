# P1 ③ 跨库独立复现 —— CPT 第二全球库（真实数据，2026-05-30）

承接 `RESULTS_P1_baseline_2026-05-29.md`。本轮拿到**真正的第二个独立全球库**并跑通。

## 数据（真实、可引用）
- **Geyin & Maurer (2021)**, CPT-Based Liquefaction Case Histories from Global Earthquakes, **DOI 10.17603/ds2-wftt-mv37**（DesignSafe PRJ-3012）。
- 用户提供 `PRJ-3012.zip` → `GLOBALDATASETV1.mat`（MATLAB v7.3，14.3 MB，`mat73` 读取）。275 例 / 21 次地震；manifestation 159 yes(coded 2) / 116 no(0)；qc[kPa]、depth/GWT[m]、PGA[g]；用 inverse-filtered "true" 剖面。provenance 见 `data/raw/geyin_maurer_2021_cpt/SOURCE.md`。

## 新建零样本 CPT 触发引擎 `geoliq/mechanics/cpt_profile.py`（不拟合标签）
Robertson(2010) 容重 → Robertson(2009) Ic 迭代 → FC(Ic, BI2014) → 有效应力(含水位) → BI2014-CPT CRR 逐层 FS → ① 临界层 FS；② **LPI（Iwasaki et al. 1978）**作"触发→地表喷冒"桥梁。
- 诚实修正：min-FS-over-profile 过度触发（FAR 0.92，深剖面总有薄弱层）→ 改用标准 **LPI**（深度加权）。

## 结果（vs 地表喷冒 manifestation）
| 量 | 值 |
|---|---|
| **BI2014-CPT + LPI 零样本 AUC** | **0.750** |
| acc(LPI≥5) | 0.713；FAR 0.534；miss 0.107 |
| ML hist-GBT | random 0.725 → 震群分组 0.613（缺口 **0.112**） |
| ML logistic | random 0.712 → 震群分组 0.648（缺口 0.064） |

产物 `data/processed/geyin2021_cpt_results.json` / `geyin2021_cpt_records.csv`。

## 跨库总结（两个独立全球库 × 两种原位试验 SPT/CPT × 两种标签 触发/喷冒）
| 库 | 试验/标签 | 物理零样本 AUC | ML 震群分组 AUC | ML 是否超物理 |
|---|---|---|---|---|
| Cetin 2018 | SPT / 触发 | **0.923** | 0.85 | 否 |
| Geyin–Maurer 2021 | CPT / 喷冒(LPI) | **0.750** | 0.61–0.65 | 否 |

1. **零样本机理在两个独立库都最强、且都没被 ML 超过**。
2. **乐观性缺口在两个库都复现**（SPT ~0.05，CPT 0.06–0.11）。
3. 机理因"不拟合"而跨库可迁移；拟合黑箱随库/分组退化。
4. 两个 AUC 量级不同是因为**标签不同**（触发 vs 地表喷冒，后者本就更难），非机理"失效"。

→ 单库结论（"诚实验证下 ML 不超规范"）已升级为**跨库、跨试验、跨标签的稳健证据**。这是 P1 冲高刊的核心卖点之一。

## 复现
```
python code\run_baseline_cpt_geyin.py
$env:PYTHONPATH="...\code"; python -m pytest code\tests -q   # 18 passed
```

## 下一步
④ conformal 可靠性带（规范/LPI 预测的保形区间 + 分组漂移下覆盖率衰减）；⑤ 出图（两库乐观性缺口、ROC、水位反事实、LPI 校准）→ 起草 P1。
