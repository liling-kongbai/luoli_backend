from math import log, sqrt


def compute_uct_score(
    child_visit_count: int,
    child_score_count: int,
    parent_visit_count: int,
    exploration_c: float = 1.414,
) -> float:
    """计算 UCT 分数"""

    if child_visit_count == 0:
        return float('inf')

    exploitation_score = child_score_count / child_visit_count  # 利用分数
    exploration_score = exploration_c * sqrt(
        log(parent_visit_count) / child_visit_count
    )  # 探索分数
    return exploitation_score + exploration_score
