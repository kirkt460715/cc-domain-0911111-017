"""Deterministic regression math on exact fractions.

Standardization and model fitting use only the rows handed to each function,
so callers control leakage by construction. Square roots are the only
operation that leaves exact arithmetic: they pass through an IEEE double and
back into a bounded-denominator fraction, which is deterministic.
"""
from fractions import Fraction
from ..values import reject


def fraction_sqrt(value):
    value = Fraction(value)
    if value < 0:
        reject("input", "square root of negative value")
    if value == 0:
        return Fraction(0)
    return Fraction(float(value) ** 0.5).limit_denominator(10**9)


def mean(column):
    if not column:
        reject("input", "mean of empty column")
    return sum(column, Fraction(0)) / len(column)


def variance(column):
    center = mean(column)
    return sum((value - center) ** 2 for value in column) / len(column)


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        reject("input", "correlation requires paired columns")
    left_center = mean(left)
    right_center = mean(right)
    left_dev = [value - left_center for value in left]
    right_dev = [value - right_center for value in right]
    left_ss = sum(value * value for value in left_dev)
    right_ss = sum(value * value for value in right_dev)
    if left_ss == 0 or right_ss == 0:
        return Fraction(0)
    numerator = sum(a * b for a, b in zip(left_dev, right_dev))
    return numerator / fraction_sqrt(left_ss * right_ss)


def standardize(columns):
    """Center and scale each column; constant columns get scale None."""
    means = []
    scales = []
    result = []
    for column in columns:
        center = mean(column)
        spread = fraction_sqrt(variance(column))
        means.append(center)
        if spread == 0:
            scales.append(None)
            result.append([Fraction(0) for _ in column])
        else:
            scales.append(spread)
            result.append([(value - center) / spread for value in column])
    return means, scales, result


def apply_standardization(columns, means, scales):
    if len(columns) != len(means):
        reject("input", "standardization width mismatch")
    result = []
    for column, center, spread in zip(columns, means, scales):
        if spread is None:
            result.append([Fraction(0) for _ in column])
        else:
            result.append([(value - center) / spread for value in column])
    return result


def solve(system, target):
    """Exact Gaussian elimination with partial pivoting."""
    size = len(system)
    if size == 0 or any(len(row) != size for row in system) or len(target) != size:
        reject("input", "linear system shape mismatch")
    matrix = [list(row) + [target[index]] for index, row in enumerate(system)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if matrix[pivot][column] == 0:
            reject("model", "singular linear system")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        for row in range(column + 1, size):
            factor = matrix[row][column] / matrix[column][column]
            for cell in range(column, size + 1):
                matrix[row][cell] -= factor * matrix[column][cell]
    result = [Fraction(0)] * size
    for row in range(size - 1, -1, -1):
        residual = matrix[row][size] - sum(
            matrix[row][column] * result[column] for column in range(row + 1, size)
        )
        result[row] = residual / matrix[row][row]
    return result


def ridge_fit(scaled_columns, scaled_target, penalty):
    """Ridge on standardized columns; penalty is an exact fraction."""
    width = len(scaled_columns)
    if width == 0:
        reject("model", "ridge requires at least one feature")
    size = len(scaled_target)
    if any(len(column) != size for column in scaled_columns) or size == 0:
        reject("input", "ridge row count mismatch")
    gram = [
        [
            sum(scaled_columns[a][i] * scaled_columns[b][i] for i in range(size))
            + (Fraction(penalty) if a == b else Fraction(0))
            for b in range(width)
        ]
        for a in range(width)
    ]
    projection = [
        sum(scaled_columns[a][i] * scaled_target[i] for i in range(size))
        for a in range(width)
    ]
    return solve(gram, projection)


def ols_univariate(feature, target):
    """Univariate least squares with intercept; flat feature gives the mean."""
    if len(feature) != len(target) or not feature:
        reject("input", "regression row count mismatch")
    feature_mean = mean(feature)
    target_mean = mean(target)
    feature_ss = sum((value - feature_mean) ** 2 for value in feature)
    if feature_ss == 0:
        return target_mean, Fraction(0)
    covariance = sum(
        (x - feature_mean) * (y - target_mean) for x, y in zip(feature, target)
    )
    slope = covariance / feature_ss
    return target_mean - slope * feature_mean, slope


def rmse(residuals):
    if not residuals:
        reject("input", "error metric of empty residuals")
    return fraction_sqrt(sum(value * value for value in residuals) / len(residuals))


def metrics(actual, predicted):
    """Goodness-of-fit against actuals; returns exact fractions."""
    if len(actual) != len(predicted) or not actual:
        reject("input", "metric row count mismatch")
    residuals = [a - p for a, p in zip(actual, predicted)]
    center = mean(actual)
    error = rmse(residuals)
    return {
        "residuals": residuals,
        "rmse": error,
        "cvrmse": error / center if center else None,
        "nmbe": (sum(residuals, Fraction(0)) / len(residuals)) / center
        if center
        else None,
        "mean_actual": center,
    }


def forward_chains(size, folds):
    """Expanding-window splits; every train window strictly precedes validation."""
    if size < 2:
        reject("model", "forward validation requires at least two rows")
    folds = max(1, min(folds, size - 1))
    chains = []
    for fold in range(1, folds + 1):
        cut = size * fold // (folds + 1)
        following = size * (fold + 1) // (folds + 1)
        if cut < 1 or following <= cut:
            continue
        chains.append((list(range(cut)), list(range(cut, following))))
    if not chains:
        reject("model", "forward validation cannot split rows")
    return chains
