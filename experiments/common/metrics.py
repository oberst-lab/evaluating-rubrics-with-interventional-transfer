"""All statistics in the paper, as pure functions of the stored judge scores.

A judge score is a map: criterion number -> 0 or 1.
No function here calls a model. Thus, the bootstrap can calculate a table many times.
"""

from statistics import fmean as mean  # fmean is much faster than statistics.mean

ScoreMap = dict[int, int]


def fraction(score_map: ScoreMap) -> float:
    """The score of a rubric on a response: the fraction of criteria that the response satisfies."""
    return mean(score_map.values())


def pairwise_differences(
    by_label: dict[str, float | None], order: list[str]
) -> dict[tuple[str, str], float | None]:
    """(later, earlier) -> by_label[later] - by_label[earlier], for each pair in `order`."""
    differences = {}
    for j, later in enumerate(order):
        for earlier in order[:j]:
            if by_label[later] is None or by_label[earlier] is None:
                differences[(later, earlier)] = None
                continue
            differences[(later, earlier)] = by_label[later] - by_label[earlier]
    return differences


def rank(xs: list[float]) -> list[float]:
    """Ranks. Equal values get the mean of their ranks."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation. None when one side is constant."""
    # Examine the values, not var == 0. A rounding error can give a constant vector a variance of 1e-33.
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return None
    mean_x, mean_y = mean(xs), mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of the ranks. None when one side is constant."""
    return pearson(xs=rank(xs=xs), ys=rank(xs=ys))


def rubricrag_correlation(
    generated: list[ScoreMap], expert: list[ScoreMap]
) -> float | None:
    """The RubricRAG score correlation (their Table 3).

    There is one response for each example. The Spearman correlation is across the examples.
    genrubric_correlation is different. It calculates the correlation in each example, across a panel of responses.
    """
    xs = [fraction(score_map=s) for s in generated]
    ys = [fraction(score_map=s) for s in expert]
    return spearman(xs=xs, ys=ys)


def genrubric_correlation(
    generated: list[list[ScoreMap]], expert: list[list[ScoreMap]]
) -> float | None:
    """The GenRubric score correlation (their App B.1, "sorting consistency").

    For each example, calculate the Spearman correlation between the two rubrics across the panel. Then calculate the mean over the examples.
    generated[i][k] and expert[i][k] are the scores of the two rubrics on response k of example i.
    If a rubric gives the same score to all of the panel, the correlation is not defined. That example is not in the mean.
    None when no example has a correlation.
    """
    per_example = [
        rubricrag_correlation(generated=by_response, expert=expert_by_response)
        for by_response, expert_by_response in zip(generated, expert)
    ]
    defined = [value for value in per_example if value is not None]
    return mean(defined) if defined else None


def preference_alignment(good: list[ScoreMap], bad: list[ScoreMap]) -> float | None:
    """The RubricRAG good-over-bad accuracy (their Section 3.3.3 ii, Figure 8).

    This is the fraction of examples where a rubric gives the physician response a higher score than the bad response.
    An equal score counts as a failure.
    This metric uses one rubric at a time. Thus, the expert rubric also gets a value.
    None when there are no examples.
    """
    if not good:
        return None
    return mean(
        float(fraction(score_map=g) > fraction(score_map=b)) for g, b in zip(good, bad)
    )


def it_impact_share(
    after: list[ScoreMap], before: list[ScoreMap], headroom: list[int]
) -> float | None:
    """The interventional transfer Q: the criteria that changed, divided by the criteria that could change.

    `headroom` is one count for each example. The caller selects it.
    For a "pass" intervention, the headroom is the failed criteria. For a "fail" intervention, it is the passed criteria.
    A fraction lets you compare rubrics of different lengths.
    The function adds all examples before the division. An example with zero headroom thus has zero weight.
    None when the total headroom is zero.
    """
    available = sum(headroom)
    if available == 0:
        return None
    return (
        sum(sum(a.values()) - sum(b.values()) for a, b in zip(after, before))
        / available
    )


def framing_precision_recall(
    matched: dict[int, list[int]], n_true: int
) -> tuple[float, float]:
    """The matching precision and recall of one example.

    `matched` is: generated criterion number -> the expert criteria it matches.
    Precision is the fraction of generated criteria that match one or more expert criteria.
    Recall is the fraction of expert criteria that one or more generated criteria match.
    """
    precision = sum(1 for row in matched.values() if row) / len(matched)
    recall = len(set().union(*matched.values())) / n_true
    return precision, recall


def framing_f1(matched: dict[int, list[int]], n_true: int) -> float:
    """The matching F1 of one example. The caller calculates the mean over examples, as rubric_rag_f1 does.

    Zero when nothing matches.
    """
    precision, recall = framing_precision_recall(matched=matched, n_true=n_true)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rubricrag_precision_recall(
    grid: dict[int, dict[int, int]], n_true: int
) -> tuple[float, float]:
    """The RubricRAG LLM-matching precision and recall of one example.

    `grid` is: generated criterion number -> expert criterion number -> similarity from 0 to 9.
    Precision is the mean of the best similarity of each generated criterion. Recall is the same for each expert criterion.
    We divide by 9. Their paper does not give the scale, but this is the only scale where precision can be 1.
    """
    precision = mean(max(by_true.values()) for by_true in grid.values()) / 9
    recall = (
        mean(
            max(by_true[true_index] for by_true in grid.values())
            for true_index in range(n_true)
        )
        / 9
    )
    return precision, recall
