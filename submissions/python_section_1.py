from typing import Dict, List, Tuple
from haversine import haversine
from datetime import datetime
import pandas as pd
import re
import polyline
import pandas as pd
import numpy as np


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    if n <= 1:
        return lst

    for i in range(0, len(lst), n):
        left = i
        right = min(i + n - 1, len(lst) - 1)

        while left < right:
            # XOR swap algorithm
            if lst[left] != lst[right]:  # perform XOR swap when elements are different
                lst[left] = lst[left] ^ lst[right]
                lst[right] = lst[left] ^ lst[right]
                lst[left] = lst[left] ^ lst[right]
            left += 1
            right -= 1

    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for s in lst:
        length = len(s)

        if length not in result:
            result[length] = []
        result[length].append(s)

    return dict(sorted(result.items()))


def flatten_dict(nested_dict: Dict, sep: str = ".") -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """

    # Added a nested function to preserve the main outer function template format
    def _flatten(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        items: Dict = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.update(_flatten(v, new_key, sep))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, (dict, list)):
                        items.update(_flatten({f"{new_key}[{i}]": item}, "", sep))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = v
        return items

    return _flatten(nested_dict, sep=sep)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """

    def should_swap(arr: List[int], start: int, curr: int) -> bool:
        for i in range(start, curr):
            if arr[i] == arr[curr]:
                return False
        return True

    def generate_unique_permutations(
        arr: List[int], start: int, result: List[List[int]]
    ) -> None:
        if start == len(arr):
            result.append(arr[:])
            return

        for i in range(start, len(arr)):
            # Skip if position already used the value
            if should_swap(arr, start, i):
                arr[start], arr[i] = arr[i], arr[start]
                generate_unique_permutations(arr, start + 1, result)
                arr[start], arr[i] = arr[i], arr[start]

    nums.sort()
    result = []
    generate_unique_permutations(nums, 0, result)
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.

    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    def is_valid_date(date_str: str, fmt: str) -> bool:
        """Helper function to validate dates"""
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            return False

    # potential date formats
    patterns = {
        r"\b(\d{2})-(\d{2})-(\d{4})\b": "%d-%m-%Y",  # dd-mm-yyyy
        r"\b(\d{2})/(\d{2})/(\d{4})\b": "%m/%d/%Y",  # mm/dd/yyyy
        r"\b(\d{4})\.(\d{2})\.(\d{2})\b": "%Y.%m.%d",  # yyyy.mm.dd
    }

    valid_dates = []

    # Find and validate each date format
    for pattern, date_format in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            date_str = match.group(0)
            if is_valid_date(date_str, date_format):
                valid_dates.append(date_str)

    return valid_dates


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    polyline_df = pd.DataFrame(coordinates, columns=["latitude", "longitude"])
    polyline_df["distance"] = [0] + [
        round(
            haversine(coordinates[i - 1], coordinates[i]) * 1000, 2
        )  # meters to two decimals
        for i in range(1, len(coordinates))
    ]
    return polyline_df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element
    by the sum of its original row and column index before rotation.

    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.

    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    transformed_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(row[j] for row in rotated) - rotated[i][j]
            transformed_matrix[i][j] = row_sum + col_sum
    return transformed_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-1 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Pre compute day to minutes mapping to reduce repeated conversions
    DAY_MINUTES = 24 * 60
    WEEK_MINUTES = 7 * DAY_MINUTES
    day_to_minutes: Dict[str, int] = {
        "Monday": 0,
        "Tuesday": DAY_MINUTES,
        "Wednesday": 2 * DAY_MINUTES,
        "Thursday": 3 * DAY_MINUTES,
        "Friday": 4 * DAY_MINUTES,
        "Saturday": 5 * DAY_MINUTES,
        "Sunday": 6 * DAY_MINUTES,
    }

    @np.vectorize
    def time_to_minutes(time_str: str) -> int:
        """Vectorized conversion of time string to minutes"""
        h, m, s = map(int, time_str.split(":"))
        return h * 60 + m + (1 if s > 0 else 0)  # Round up seconds to next minute

    def create_time_ranges(group: pd.DataFrame) -> List[Tuple[int, int]]:
        """Convert a group's time ranges to minute intervals"""
        ranges = []

        # Vectorized operations for start and end times
        start_minutes = time_to_minutes(group["startTime"].values)
        end_minutes = time_to_minutes(group["endTime"].values)

        # Convert days to minute offsets
        start_day_minutes = np.array([day_to_minutes[day] for day in group["startDay"]])
        end_day_minutes = np.array([day_to_minutes[day] for day in group["endDay"]])

        # Total start and end minutes
        starts = start_day_minutes + start_minutes
        ends = end_day_minutes + end_minutes

        # Handle week wraparound
        mask = ends < starts
        ends[mask] += WEEK_MINUTES

        # Create ranges
        for start, end in zip(starts, ends):
            ranges.append((start % WEEK_MINUTES, end % WEEK_MINUTES))

        return ranges

    def check_coverage(ranges: List[Tuple[int, int]]) -> bool:
        """
        Check if ranges cover the full week using an optimized sweep line algorithm
        """
        if not ranges:
            return True  # Empty ranges mean incomplete coverage

        # Sort ranges by start time
        ranges.sort()

        # Early exit if first range starts after 0 or last range ends before week end
        if ranges[0][0] > 0:
            return True

        # Merge overlapping ranges
        merged = [ranges[0]]
        for curr_start, curr_end in ranges[1:]:
            prev_end = merged[-1][1]

            # Handle week wraparound
            if curr_start > curr_end:
                curr_end += WEEK_MINUTES

            if curr_start <= prev_end:
                merged[-1] = (merged[-1][0], max(prev_end, curr_end))
            else:
                # A time gap has been found, so it's incomplete
                return True

        # Check if the merged ranges cover the full week
        return merged[0][0] > 0 or merged[-1][1] < WEEK_MINUTES

    # Process each group
    results = {}

    # Group data for vectorized operations
    for (id_val, id_2_val), group in df.groupby(["id", "id_2"]):
        # Quick check for minimum required coverage
        if len(group) < 7:  # Need at least 7 entries for full week coverage
            results[(id_val, id_2_val)] = True
            continue

        ranges = create_time_ranges(group)
        results[(id_val, id_2_val)] = check_coverage(ranges)

    # Create series with properly formatted index
    series = pd.Series(results, name="has_incorrect_timestamps")

    # Convert MultiIndex to single index with formatted strings
    series.index = [f"{id_val}   {id_2_val}" for id_val, id_2_val in series.index]

    return series
