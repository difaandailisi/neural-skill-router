#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经形态能力调度器 (Neural Morphological Skill Router)
=====================================================
把"意图驱动的能力装卸"闭环实现为可运行算法。

神经机制 -> 算法组件:
  场景线索      -> encode()         意图编码
  树突整合      -> score()          匹配评分
  侧向抑制      -> kwta()           稀疏激活 top-k
  赫布学习      -> hebbian()        用进废退 + 髓鞘化
  突触修剪      -> decay_prune()    衰减归还
  动态扩展      -> discover()       搜索并租借新能力

无外部依赖，Python 3.6+ 直接运行。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

# ---------------------------------------------------------------- 分词
STOP = set("的了和是在我你帮给个一下把用于请这那有")


def tokenize(text: str) -> Set[str]:
    """中英文混合分词：英文按词，中文只用 bigram。
    不用单字是为了避免假匹配——例如「描写」的「描」和「扫描」的「描」
    若都作为单字 token，会让写作任务误点亮蓝牙扫描能力。"""
    text = text.lower()
    tokens: Set[str] = set(re.findall(r"[a-z0-9]+", text))
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.add(seg)                       # 单字片段只能保留单字
        for i in range(len(seg) - 1):
            tokens.add(seg[i:i + 2])              # 二字组是中文的最小语义单位
    # STOP 里的单字只用于过滤单字片段
    return {t for t in tokens if t not in STOP}


def dice(a: Set[str], b: Set[str]) -> float:
    """Dice 系数：集合重叠度，0~1。"""
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


# ---------------------------------------------------------------- 数据结构
@dataclass
class Capability:
    name: str
    kind: str                       # skill / mcp / plugin
    keywords: Set[str]
    revocation_cost: float = 0.2    # 撤销成本 [0,1]：低=>可自动租借，高=>需确认
    base_weight: float = 0.35       # 初始突触强度
    weight: float = field(default=None)
    activation: float = 0.0
    borrowed: bool = False          # 是否临时租借（未永久安装）
    frozen: bool = False            # 是否髓鞘化固化
    last_used: int = -1
    use_count: int = 0

    def __post_init__(self):
        if self.weight is None:
            self.weight = self.base_weight


@dataclass
class Config:
    top_k: int = 2                  # 一次最多点亮几个能力（稀疏度）
    eta: float = 0.30               # Hebbian 学习率
    decay: float = 0.12             # 每轮衰减率（用进废退）
    freeze_after: int = 4           # 使用几次后髓鞘化冻结
    prune_after: int = 2            # 借来的能力闲置几轮后归还
    switch_threshold: float = 0.12  # 上下文相似度低于此值判定"换任务"
    borrow_cost_limit: float = 0.4  # 撤销成本低于此值可自动租借
    min_match: float = 0.08         # 触发发现流程的最低匹配度
    dedup_threshold: float = 0.75   # 关键词重复度高于此值视为重复能力


# ---------------------------------------------------------------- 调度器
class NeuralRouter:
    def __init__(self, installed: List[Capability], catalog: List[Capability], cfg: Config):
        self.installed = installed          # 当前已挂载的能力（含借入的）
        self.catalog = catalog              # 可搜索的能力市场（未安装）
        self.cfg = cfg
        self.round = 0
        self.last_intent: Set[str] = set()

    def _by_name(self, names: List[str]) -> List[Capability]:
        return [c for c in self.installed if c.name in names]

    # 1. 意图编码 ----------------------------------------------------
    def encode(self, text: str) -> Set[str]:
        return tokenize(text)

    # 2. 匹配评分（树突整合）-----------------------------------------
    def score(self, intent: Set[str], cap: Capability) -> float:
        overlap = dice(intent, cap.keywords)
        # Hebbian 权重参与门控：常用能力更容易被点亮（熟练度红利）
        return overlap * (0.5 + cap.weight)

    # 3. k-WTA 稀疏激活（侧向抑制）-----------------------------------
    def kwta(self, scores: Dict[str, float]) -> List[str]:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, val in ranked[:self.cfg.top_k] if val > 0]

    # 任务边界：上下文相似度骤降 -> 切换 ----------------------------
    def is_switch(self, intent: Set[str]) -> bool:
        if not self.last_intent:
            return False
        return dice(intent, self.last_intent) < self.cfg.switch_threshold

    # 4+5. 发现/去重/授权分级 ---------------------------------------
    def discover(self, intent: Set[str]) -> tuple:
        borrowed, pending, skipped = [], [], []
        for cand in list(self.catalog):
            if any(c.name == cand.name for c in self.installed):
                continue                              # 已挂载，跳过
            ov = dice(intent, cand.keywords)
            if ov < self.cfg.min_match:
                continue
            if any(dice(cand.keywords, c.keywords) >= self.cfg.dedup_threshold
                   for c in self.installed):
                skipped.append(cand.name)             # 去重：已有等价能力
                continue
            if cand.revocation_cost <= self.cfg.borrow_cost_limit:
                cand.borrowed = True
                self.installed.append(cand)
                self.catalog.remove(cand)
                borrowed.append(cand.name)            # 低撤销成本 => 自动租借
            else:
                pending.append(cand.name)             # 高撤销成本 => 请求确认
        return borrowed, pending, skipped

    # 6. Hebbian 强化（髓鞘化）---------------------------------------
    def hebbian(self, winners: List[str], scores: Dict[str, float]):
        for cap in self._by_name(winners):
            cap.weight += self.cfg.eta * scores[cap.name] * (1 - cap.weight)
            cap.activation = scores[cap.name]
            cap.last_used = self.round
            cap.use_count += 1
            if cap.use_count >= self.cfg.freeze_after:
                cap.frozen = True                     # 固化：不再衰减

    # 7. 衰减 + 归还（突触修剪）--------------------------------------
    def decay_prune(self, winners: List[str], switched: bool) -> List[str]:
        pruned = []
        for cap in self.installed:
            if cap.name in winners:
                continue
            cap.activation = 0.0
            if cap.frozen:
                continue                              # 髓鞘化通路抗衰减
            factor = self.cfg.decay * (2 if switched else 1)
            cap.weight *= (1 - factor)
            if (cap.borrowed and cap.last_used >= 0
                    and self.round - cap.last_used >= self.cfg.prune_after):
                self.installed.remove(cap)            # 闲置够久 => 归还
                self.catalog.append(cap)
                cap.borrowed = False
                pruned.append(cap.name)
        return pruned

    # 单轮推进 -------------------------------------------------------
    def step(self, text: str) -> dict:
        self.round += 1
        intent = self.encode(text)
        switched = self.is_switch(intent)

        borrowed, pending, skipped = self.discover(intent)

        scores = {c.name: self.score(intent, c) for c in self.installed}
        winners = self.kwta(scores)

        self.hebbian(winners, scores)
        pruned = self.decay_prune(winners, switched)

        self.last_intent = intent
        return {
            "round": self.round, "intent": text, "switched": switched,
            "borrowed": borrowed, "pending": pending, "skipped": skipped,
            "winners": winners, "pruned": pruned,
        }

    def snapshot(self) -> List[str]:
        rows = []
        for c in sorted(self.installed, key=lambda x: -x.weight):
            flag = ("F" if c.frozen else ".") + ("B" if c.borrowed else ".")
            rows.append(f"{c.name:22s} {c.kind:6s} w={c.weight:.3f} "
                        f"act={c.activation:.2f} [{flag}]")
        return rows


# ---------------------------------------------------------------- Demo
def build_demo() -> NeuralRouter:
    installed = [
        Capability("ble-battery-mcp", "mcp",
                   tokenize("BLE 蓝牙 电量 battery 协议 protocol 扫描 scan"), 0.8),
        Capability("excel-pivot-skill", "skill",
                   tokenize("Excel 透视 表格 数据 pivot sheet 汇总"), 0.7),
        Capability("code-review-mcp", "mcp",
                   tokenize("代码 review 审查 bug 修复 code"), 0.9),
    ]
    catalog = [
        Capability("creative-writing-skill", "skill",
                   tokenize("小说 描写 写作 故事 文风 短篇 场景 write novel"), 0.15),
        Capability("obsidian-vault-mcp", "mcp",
                   tokenize("obsidian 笔记 记录 记下 vault markdown md"), 0.25),
        Capability("deep-research-skill", "skill",
                   tokenize("调研 研究 research 检索 文献 报告"), 0.35),
    ]
    return NeuralRouter(installed, catalog, Config())


def main():
    router = build_demo()
    turns = [
        "帮我写一部小说的短篇描写，用 Obsidian 记下来",
        "再写一段雨景的描写，也存进笔记",
        "继续，把结尾再描写得细一点",
        "查一下 BLE 电量协议的 bug",
        "帮我做个 Excel 数据透视",
    ]

    print("=" * 68)
    print("神经形态能力调度器 · 演示")
    print("=" * 68)
    for t in turns:
        r = router.step(t)
        print(f"\n[轮 {r['round']}] {r['intent']}")
        print(f"  任务切换       : {'是' if r['switched'] else '否'}")
        print(f"  自动租借 (+)   : {r['borrowed'] or '—'}")
        print(f"  待你确认 (?)   : {r['pending'] or '—'}")
        print(f"  去重跳过 (=)   : {r['skipped'] or '—'}")
        print(f"  点亮 (k-WTA)   : {r['winners'] or '—'}")
        print(f"  归还/修剪 (-)  : {r['pruned'] or '—'}")
    print("\n" + "-" * 68)
    print("最终状态   [F=已髓鞘化  B=临时借入]")
    for row in router.snapshot():
        print("  " + row)
    print("-" * 68)


if __name__ == "__main__":
    main()
