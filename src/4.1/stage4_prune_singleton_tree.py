"""Stage 4: 自底向上裁剪单样本簇并压缩单子节点父节点。

输入：
- stage3 产出的 capability_tree_final.json

处理规则：
1) 自底向上删除“叶子且仅 1 条数据”的簇。
2) 若删除后父节点仅剩 1 个子节点，则执行“子替父”折叠。
3) 重新计算 subtree_size / children_count / leaf_payload_size。

输出：
- 新的 capability_tree_final（裁剪后）
- 新的 capability_tree_summary（裁剪统计）
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PruneStats:
    removed_singleton_leaves: int = 0
    removed_empty_nodes: int = 0
    collapsed_single_child_parents: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: Prune singleton clusters from stage3 tree")
    parser.add_argument(
        "--input-tree-json",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/capability_tree_final.json"),
    )
    parser.add_argument(
        "--output-tree-json",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/capability_tree_final_pruned.json"),
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/capability_tree_summary_pruned.json"),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def normalize_node(node: dict[str, Any]) -> dict[str, Any]:
    """确保节点具备最小字段，避免旧版本 JSON 缺键时报错。"""
    node.setdefault("node_id", "UNKNOWN")
    node.setdefault("data_ids", [])
    node.setdefault("children", [])
    node.setdefault("center_norm", 0.0)
    if not isinstance(node["data_ids"], list):
        node["data_ids"] = []
    if not isinstance(node["children"], list):
        node["children"] = []
    for c in node["children"]:
        normalize_node(c)
    return node


def collect_subtree_ids(node: dict[str, Any]) -> list[str]:
    ids = list(node.get("data_ids", []))
    for c in node.get("children", []):
        ids.extend(collect_subtree_ids(c))
    return list(dict.fromkeys(str(x) for x in ids))


def recalc_node_fields(node: dict[str, Any]) -> None:
    for c in node.get("children", []):
        recalc_node_fields(c)
    node["leaf_payload_size"] = len(node.get("data_ids", []))
    node["children_count"] = len(node.get("children", []))
    node["subtree_size"] = len(collect_subtree_ids(node))


def prune_bottom_up(node: dict[str, Any], stats: PruneStats) -> dict[str, Any] | None:
    """返回裁剪后的节点；若整棵子树被删空则返回 None。"""
    children = node.get("children", [])
    pruned_children: list[dict[str, Any]] = []
    for child in children:
        out = prune_bottom_up(child, stats)
        if out is not None:
            pruned_children.append(out)
    node["children"] = pruned_children

    is_leaf = len(node["children"]) == 0
    payload = len(node.get("data_ids", []))

    # 规则 1：删除单样本叶子簇
    if is_leaf and payload == 1:
        stats.removed_singleton_leaves += 1
        return None

    # 清理空节点
    if is_leaf and payload == 0:
        stats.removed_empty_nodes += 1
        return None

    # 规则 2：父节点仅剩 1 个子节点 -> 子替父
    if len(node["children"]) == 1:
        stats.collapsed_single_child_parents += 1
        return node["children"][0]

    return node


def count_nodes(node: dict[str, Any]) -> int:
    return 1 + sum(count_nodes(c) for c in node.get("children", []))


def depth(node: dict[str, Any]) -> int:
    if not node.get("children"):
        return 0
    return 1 + max(depth(c) for c in node["children"])


def level_counts(node: dict[str, Any]) -> dict[int, int]:
    counts: dict[int, int] = {}

    def _walk(n: dict[str, Any], lv: int) -> None:
        counts[lv] = counts.get(lv, 0) + 1
        for c in n.get("children", []):
            _walk(c, lv + 1)

    _walk(node, 0)
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def singleton_leaf_count(node: dict[str, Any]) -> int:
    if not node.get("children"):
        return 1 if len(node.get("data_ids", [])) == 1 else 0
    return sum(singleton_leaf_count(c) for c in node["children"])


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_tree_json.exists():
        raise FileNotFoundError(f"Input tree not found: {args.input_tree_json}")

    with args.input_tree_json.open("r", encoding="utf-8") as f:
        raw_tree = json.load(f)
    tree = normalize_node(raw_tree)

    before_nodes = count_nodes(tree)
    before_depth = depth(tree)
    before_level_counts = level_counts(tree)
    before_singletons = singleton_leaf_count(tree)
    before_subtree_size = len(collect_subtree_ids(tree))

    stats = PruneStats()
    pruned = prune_bottom_up(tree, stats)

    if pruned is None:
        # 全部被剪空时，输出一个空根节点，避免后续流程读不到树结构。
        pruned = {
            "node_id": "EMPTY_ROOT",
            "center_norm": 0.0,
            "data_ids": [],
            "children": [],
        }

    recalc_node_fields(pruned)

    after_nodes = count_nodes(pruned)
    after_depth = depth(pruned)
    after_level_counts = level_counts(pruned)
    after_singletons = singleton_leaf_count(pruned)
    after_subtree_size = len(collect_subtree_ids(pruned))

    args.output_tree_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    with args.output_tree_json.open("w", encoding="utf-8") as f:
        json.dump(pruned, f, ensure_ascii=False, indent=2)

    summary = {
        "input_tree_json": str(args.input_tree_json),
        "output_tree_json": str(args.output_tree_json),
        "before": {
            "total_nodes": before_nodes,
            "depth": before_depth,
            "level_counts": before_level_counts,
            "singleton_leaf_count": before_singletons,
            "subtree_size": before_subtree_size,
        },
        "after": {
            "total_nodes": after_nodes,
            "depth": after_depth,
            "level_counts": after_level_counts,
            "singleton_leaf_count": after_singletons,
            "subtree_size": after_subtree_size,
        },
        "prune_actions": {
            "removed_singleton_leaves": stats.removed_singleton_leaves,
            "removed_empty_nodes": stats.removed_empty_nodes,
            "collapsed_single_child_parents": stats.collapsed_single_child_parents,
        },
    }
    with args.output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Stage4 prune done.")
    logging.info("Saved pruned tree: %s", args.output_tree_json)
    logging.info("Saved pruned summary: %s", args.output_summary_json)
    logging.info("Before nodes=%d, after nodes=%d", before_nodes, after_nodes)


if __name__ == "__main__":
    main()
