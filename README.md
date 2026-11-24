# neural-skill-router

神经形态能力调度器：把「意图驱动的能力装卸」闭环写成可运行算法。

> 给 AI 的一句话版本：这是一个 **决策层原型**（MoE gate），不是 WorkBuddy 插件、不是 MCP server、不是可安装技能。它输入用户自然语言，输出"该点亮哪些能力 / 该租借谁 / 该归还谁"，**不执行真实的启用或关闭**。

## 它解决什么问题

用户想要的能力管理不是静态清单，而是一个闭环：

1. 从自然语言判断任务类型与目标载体（写小说 → 写作用技能；"用 Obsidian 写" → 笔记 MCP）
2. 盘点当前已挂载能力，发现缺口
3. 搜索候选能力，去重、评分、判断互补性
4. 低撤销成本的能力自动"租借"，高成本的挂起等确认
5. 用进废退：常用能力强化（髓鞘化），闲置的借入能力自动归还
6. 话题切换时快速衰减旧上下文，不给下一个任务留残留

## 神经机制 → 算法组件

| 神经机制 | 算法 | 代码位置 |
|---|---|---|
| 场景线索触发 | 意图编码（中英混合 bigram 分词） | `tokenize()` / `encode()` |
| 树突整合 | Dice 重叠度 × 熟练度权重 | `score()` |
| 侧向抑制 / 稀疏放电 | k-Winners-Take-All，每轮只点亮 top-k | `kwta()` |
| 赫布学习 | 用得越多权重越高 | `hebbian()` |
| 髓鞘化 | 用满 4 次后冻结，不再衰减 | `Capability.frozen` |
| 突触修剪 | 闲置轮数达标后自动归还 | `decay_prune()` |
| 动态扩展 | 搜索候选池，按撤销成本分级租借/挂起 | `discover()` |
| 注意力转移 | 上下文相似度骤降判定任务切换 | `is_switch()` |

## 运行

```bash
python neural_skill_router.py
```

零第三方依赖，Python 3.6+。五轮演示会打印每轮的租借 / 点亮 / 归还轨迹和最终突触状态。

## 公开 API

```python
from neural_skill_router import NeuralRouter, Capability, Config

router = NeuralRouter(installed, catalog, Config(top_k=2))
result = router.step("帮我写一部小说的短篇描写，用 Obsidian 记下来")
# result = {round, intent, switched, borrowed, pending, skipped, winners, pruned}
```

`Capability(name, kind, keywords, revocation_cost, base_weight)`：

- `revocation_cost ∈ [0,1]`：撤销成本。≤ `borrow_cost_limit`（默认 0.4）才自动租借；更高的进 `pending` 等人工确认。
- `weight`：初始突触强度，参与门控评分；反复被点亮后按 Hebbian 公式提升，到 `freeze_after` 次后冻结。
- `borrowed`：是否为临时租借。只有借入的能力会在闲置 `prune_after` 轮后被归还。

`Config` 可调参数：`top_k`、`eta`（学习率）、`decay`、`freeze_after`、`prune_after`、`switch_threshold`、`borrow_cost_limit`、`min_match`、`dedup_threshold`。

## 已验证的行为

演示中五轮对话的实际输出：

| 轮次 | 输入 | 点亮 | 租借 | 归还 |
|---|---|---|---|---|
| 1 | 写小说短篇描写，用 Obsidian 记下来 | 写作技能、Obsidian MCP | 写作技能、Obsidian MCP | — |
| 2 | 再写一段雨景描写，存进笔记 | 写作技能、Obsidian MCP | — | — |
| 3 | 继续，把结尾再描写得细一点 | 写作技能 | — | — |
| 4 | 查一下 BLE 电量协议的 bug | BLE MCP、代码审查 MCP | — | Obsidian MCP |
| 5 | 做个 Excel 数据透视 | Excel 技能 | — | 写作技能 |

开发中抓到并修复过一个真实 bug：中文单字分词让"**描写**"的"描"和"**扫描**"的"描"成为同一个 token，导致写作任务误点亮蓝牙扫描能力。现改为只用 bigram（二字组）。

## 明确的能力边界（AI 必读）

**它做到**：意图编码、评分、top-k 稀疏选择、去重、撤销成本分级、Hebbian 强化、髓鞘化、衰减归还、任务切换检测。

**它没做到**：

- 枚举真实的技能 / MCP / 连接器状态——`installed` 和 `catalog` 是硬编码假数据
- 真实的启用 / 禁用 / 安装 / 卸载——"租借 / 归还"只是 Python 列表增删
- 读写 `mcp.json`、搜索真实技能市场、连接健康度查询、回读验证
- 互补性判断——目前只是机械点亮分数最高的两个，没判断"写作技能 + 笔记 MCP"是否真互补

所以：**这是决策层原型，不是可插拔组件，也不是能直接操作 WorkBuddy 的自动化脚本。**

详细的 AI 使用约定见 `AGENTS.md`。

## 由来

从三个问题推导出来的设计：能力太多如何不超载（稀疏激活）、什么时候关闭临时能力（衰减而非开关）、要不要先问用户（按撤销成本分级）。人类对应机制：外置、压缩、稀疏、满意、租借、懒清理。

行业对应：MoE Top-k 门控（Mixtral / GPT-4）、k-WTA 竞争抑制、Hebbian 持续学习（HebbCL）。正统但过重的参照是 Numenta HTM / NuPIC。
