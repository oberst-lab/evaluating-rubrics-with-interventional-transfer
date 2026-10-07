import random
import numpy as np


def bootstrap_cells(
    compute,
    records: list[dict],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, tuple[float, float, float] | None]]:
    """Calculate a percentile interval for each cell of the table that `compute` makes.

    `compute` takes a list of records and returns row -> column -> number.
    Each replicate draws len(records) records with replacement and calculates the full table again.
    All cells in one replicate use the same resample. Thus, you can compare the cells in a table.
    A cell can be None in a replicate. That replicate is not used for the interval of that cell.
    The resample is over records only. The intervals do not include the variance between judge samples.
    """
    point_estimates = compute(records)
    rng = random.Random(seed)
    replicates = []
    for _ in range(n_bootstrap):
        resample = [records[rng.randrange(len(records))] for _ in range(len(records))]
        replicates.append(compute(resample))

    table = {}
    for row, by_column in point_estimates.items():
        table[row] = {}
        for column, point in by_column.items():
            drawn = [
                replicate[row][column]
                for replicate in replicates
                if replicate[row][column] is not None
            ]
            if point is None or not drawn:
                table[row][column] = None
                continue
            tail = (1 - confidence) / 2
            left, right = np.percentile(drawn, [100 * tail, 100 * (1 - tail)])
            table[row][column] = (float(left), point, float(right))
    return table
