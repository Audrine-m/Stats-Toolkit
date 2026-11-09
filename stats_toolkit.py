"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"


def rejection_reason(raw_value: str) -> str | None:
    """Explain why a raw deposit value should be rejected."""

    if raw_value is None or raw_value.strip() == "":
        return "Missing value"

    cleaned_value = raw_value.strip()

    if cleaned_value.lower() in ["-", "n/a", "na", "null", "none"]:
        return "Placeholder value"

    try:
        number = float(cleaned_value)
    except ValueError:
        return "Non-numeric or incorrectly formatted value"

    if not math.isfinite(number):
        return "Non-finite numeric value"

    if number < 0:
        return "Negative deposit amount"

    return None


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Load valid numeric values and record rejected values."""

    valid_values = []
    rejected_values = []

    with open(path, "r", newline="", encoding="utf-8-sig") as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header")

        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' does not exist")

        for row in reader:

            raw_value = row.get(column)

            reason = rejection_reason(raw_value)

            if reason is not None:
                rejected_values.append((reader.line_num, raw_value))
                continue

            valid_values.append(float(raw_value.strip()))

    return valid_values, rejected_values


def mean(values: list[float]) -> float:
    """Calculate the arithmetic mean."""
    if len(values) == 0:
        raise ValueError("Cannot calculate mean of empty data")

    total = 0

    for value in values:
        total += value

    return total / len(values)


def _median_sorted(sorted_values: list[float]) -> float:
    """Calculate median from an already sorted list."""

    n = len(sorted_values)

    if n == 0:
        raise ValueError("Cannot calculate median of empty data")

    middle = n // 2

    if n % 2 == 1:
        return sorted_values[middle]

    return (
        sorted_values[middle - 1] + sorted_values[middle]
    ) / 2


def median(values: list[float]) -> float:
    """Calculate the median without modifying the original list."""

    sorted_values = sorted(values)

    return _median_sorted(sorted_values)


def mode(values: list[float]) -> list[float]:
    """Return all modes, or [] when every value occurs once."""
    if len(values) == 0:
        raise ValueError("Cannot calculate mode of empty data")

    frequencies = {}

    for value in values:
        if value in frequencies:
            frequencies[value] += 1
        else:
            frequencies[value] = 1

    highest_frequency = max(frequencies.values())

    if highest_frequency == 1:
        return []

    modes = []

    for value, frequency in frequencies.items():
        if frequency == highest_frequency:
            modes.append(value)

    return sorted(modes)


def variance(values: list[float], sample: bool = True) -> float:
    """Calculate sample or population variance."""

    n = len(values)

    if sample and n < 2:
        raise ValueError("Sample variance requires at least two values")

    if not sample and n < 1:
        raise ValueError("Population variance requires at least one value")

    average = mean(values)

    squared_differences = []

    for value in values:
        difference = value - average
        squared_difference = difference ** 2
        squared_differences.append(squared_difference)

    total = sum(squared_differences)

    if sample:
        return total / (n - 1)

    return total / n


def std_dev(values: list[float], sample: bool = True) -> float:
    """Calculate sample or population standard deviation."""

    return math.sqrt(variance(values, sample))


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Calculate quartiles using the median-of-halves method."""

    if len(values) < 2:
        raise ValueError("At least two values are required")

    sorted_values = sorted(values)

    n = len(sorted_values)
    middle = n // 2

    q2 = _median_sorted(sorted_values)

    if n % 2 == 0:
        lower_half = sorted_values[:middle]
        upper_half = sorted_values[middle:]
    else:
        lower_half = sorted_values[:middle]
        upper_half = sorted_values[middle + 1:]

    q1 = _median_sorted(lower_half)
    q3 = _median_sorted(upper_half)

    return q1, q2, q3


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Detect potential outliers using the IQR method."""

    if len(values) < 2:
        raise ValueError("At least two values are required")

    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be a positive finite number")

    q1, q2, q3 = quartiles(values)

    iqr = q3 - q1

    lower_fence = q1 - (k * iqr)
    upper_fence = q3 + (k * iqr)

    outliers = []

    for value in values:
        if value < lower_fence or value > upper_fence:
            outliers.append(value)

    return {
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers": outliers
    }


def quartiles_from_sorted(
    sorted_values: list[float]
) -> tuple[float, float, float]:
    """Calculate quartiles without sorting the data again."""

    n = len(sorted_values)

    if n < 2:
        raise ValueError("At least two values are required")

    middle = n // 2

    q2 = _median_sorted(sorted_values)

    if n % 2 == 0:
        lower_half = sorted_values[:middle]
        upper_half = sorted_values[middle:]
    else:
        lower_half = sorted_values[:middle]
        upper_half = sorted_values[middle + 1:]

    q1 = _median_sorted(lower_half)
    q3 = _median_sorted(upper_half)

    return q1, q2, q3


def format_mode(modes):
    if not modes:
        return "none"

    return ", ".join(f"{value:,.2f}" for value in modes)


class StatisticsAnalyzer:
    """Analyze a numeric dataset using our own statistical functions."""

    def __init__(
        self,
        values: list[float],
        name: str = "values",
        rejected: list[tuple[int, str]] | None = None,
        rows_read: int | None = None
    ) -> None:

        if len(values) < 2:
            raise ValueError("Analyzer requires at least two valid values")

        self.values = list(values)
        self.name = name

        # Sort once and reuse this list.
        self.sorted_values = sorted(self.values)

        # Keep track of rejected CSV records.
        self.rejected = rejected if rejected is not None else []

        self.rows_read = (
            rows_read if rows_read is not None
            else len(self.values) + len(self.rejected)
        )

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        """Create an analyzer directly from a CSV column."""

        values, rejected = load_column(path, column)

        return cls(
            values,
            name=column,
            rejected=rejected,
            rows_read=len(values) + len(rejected)
        )

    def summary(self) -> dict:
        """Return a dictionary containing the dataset statistics."""

        values = self.values
        sorted_values = self.sorted_values

        average = mean(values)
        middle = _median_sorted(sorted_values)
        modes = mode(values)

        sample_variance = variance(values, sample=True)
        sample_std = std_dev(values, sample=True)

        q1, q2, q3 = quartiles_from_sorted(sorted_values)

        iqr = q3 - q1

        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)

        outliers = [
            value for value in values
            if value < lower_fence or value > upper_fence
        ]

        return {
            "name": self.name,
            "count": len(values),
            "mean": average,
            "median": middle,
            "mode": modes,
            "variance": sample_variance,
            "std_dev": sample_std,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outliers": outliers
        }

    def without_outliers(self) -> "StatisticsAnalyzer":
        """Return a new analyzer without the detected outliers."""

        result = self.summary()

        outliers = result["outliers"]

        cleaned_values = [
            value for value in self.values
            if value not in outliers
        ]

        return StatisticsAnalyzer(
            cleaned_values,
            name=self.name,
            rejected=self.rejected,
            rows_read=self.rows_read
        )

    def report(self):
        all_summary = self.summary()
        clean_analyzer = self.without_outliers()
        clean_summary = clean_analyzer.summary()

        lines = []

        # Report heading
        lines.append(f"=== Statistics report: {self.name} ===")

        # Data quality information
        lines.append(f"Rows read         : {self.rows_read}")
        lines.append(f"Valid values      : {len(self.values)}")
        lines.append(f"Rejected values   : {len(self.rejected)}")

        # Explain rejected records
        for row_number, raw_value in self.rejected:
            reason = rejection_reason(raw_value)

            lines.append(
                f"  row {row_number}: '{raw_value}' ({reason})"
            )

        # Table heading
        lines.append("")
        lines.append(
            f"{'':20} {'all data':>15} {'outliers removed':>22}"
        )

        # Statistics to compare
        statistics = [
            ("count", "count"),
            ("mean", "mean"),
            ("median", "median"),
            ("mode", "mode"),
            ("variance (sample)", "variance"),
            ("std dev (sample)", "std_dev"),
            ("min / max", "min_max"),
            ("Q1 / Q3", "quartiles"),
            ("IQR", "iqr"),
        ]

        for label, key in statistics:

            if key == "count":
                all_value = str(all_summary["count"])
                clean_value = str(clean_summary["count"])

            elif key == "mode":
                all_value = format_mode(all_summary["mode"])
                clean_value = format_mode(clean_summary["mode"])

            elif key == "min_max":
                all_value = (
                    f"{all_summary['min']:,.2f} / "
                    f"{all_summary['max']:,.2f}"
                )
                clean_value = (
                    f"{clean_summary['min']:,.2f} / "
                    f"{clean_summary['max']:,.2f}"
                )

            elif key == "quartiles":
                all_value = (
                    f"{all_summary['q1']:,.2f} / "
                    f"{all_summary['q3']:,.2f}"
                )
                clean_value = (
                    f"{clean_summary['q1']:,.2f} / "
                    f"{clean_summary['q3']:,.2f}"
                )

            else:
                all_value = f"{all_summary[key]:,.2f}"
                clean_value = f"{clean_summary[key]:,.2f}"

            lines.append(
                f"{label:20} {all_value:>15} {clean_value:>22}"
            )

        # Outlier fences
        lines.append("")
        lines.append(
            f"Outlier fences      : lower "
            f"{all_summary['lower_fence']:,.2f}, "
            f"upper {all_summary['upper_fence']:,.2f}"
        )

        # Identify outliers with their dates and agent IDs
        outlier_details = []

        with open(DATA_PATH, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row in reader:
                raw_value = row.get(self.name)

                if rejection_reason(raw_value) is None:
                    value = float(raw_value.strip())

                    if (
                        value < all_summary["lower_fence"]
                        or value > all_summary["upper_fence"]
                    ):
                        outlier_details.append(
                            f"{value:,.2f} "
                            f"({row['date']}, {row['agent_id']})"
                        )

        lines.append(
            "Outliers            : " + (
                ", ".join(outlier_details)
                if outlier_details
                else "None"
            )
        )

        # NumPy verification
        lines.append("")
        lines.append("=== NumPy verification ===")

        numpy_results = compare_with_numpy(self)

        verification_labels = [
            ("mean", "mean"),
            ("median", "median"),
            ("variance", "variance"),
            ("std_dev", "std dev"),
            ("q1", "Q1"),
            ("q3", "Q3"),
        ]

        for key, label in verification_labels:
            status = "MATCH" if numpy_results[key] else "MISMATCH"

            if key == "variance":
                lines.append(f"{label:10} {status} (ddof=1)")
            else:
                lines.append(f"{label:10} {status}")

        lines.append("Q1 / Q3     " + (
            "MATCH" if numpy_results["q1"] and numpy_results["q3"]
            else "MISMATCH"
        ) + " (method=median of halves)")

        return "\n".join(lines)


def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Compare our calculations against NumPy."""

    import numpy as np

    values = analyzer.values

    # Our calculations
    our_mean = mean(values)
    our_median = median(values)
    our_variance = variance(values, sample=True)
    our_std = std_dev(values, sample=True)

    our_q1, our_q2, our_q3 = quartiles(values)

    # NumPy calculations
    numpy_mean = float(np.mean(values))
    numpy_median = float(np.median(values))

    # ddof=1 means sample variance and sample standard deviation
    numpy_variance = float(np.var(values, ddof=1))
    numpy_std = float(np.std(values, ddof=1))

    # Match NumPy's median-of-halves quartile method
    sorted_array = np.sort(values)

    n = len(sorted_array)
    middle = n // 2

    if n % 2 == 0:
        lower_half = sorted_array[:middle]
        upper_half = sorted_array[middle:]
    else:
        lower_half = sorted_array[:middle]
        upper_half = sorted_array[middle + 1:]

    numpy_q1 = float(np.median(lower_half))
    numpy_q2 = float(np.median(sorted_array))
    numpy_q3 = float(np.median(upper_half))

    # Compare using numerical tolerance
    tolerance = {
        "rel_tol": 1e-9,
        "abs_tol": 1e-9
    }

    results = {
        "mean": math.isclose(our_mean, numpy_mean, **tolerance),
        "median": math.isclose(our_median, numpy_median, **tolerance),
        "variance": math.isclose(our_variance, numpy_variance, **tolerance),
        "std_dev": math.isclose(our_std, numpy_std, **tolerance),
        "q1": math.isclose(our_q1, numpy_q1, **tolerance),
        "q2": math.isclose(our_q2, numpy_q2, **tolerance),
        "q3": math.isclose(our_q3, numpy_q3, **tolerance)
    }

    return results


if __name__ == "__main__":

    analyzer = StatisticsAnalyzer.from_csv(
        DATA_PATH,
        "daily_deposits_kes"
    )

    report_text = analyzer.report()

    print(report_text)

    output_path = Path("output") / "stats_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("\nReport saved successfully!")