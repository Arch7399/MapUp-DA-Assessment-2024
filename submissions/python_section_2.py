import pandas as pd
import numpy as np
from datetime import time


def calculate_distance_matrix(
    df,
) -> (
    pd.DataFrame
):  # Cannot call in type expressions, hence replacing 'pd.DataFrame(): to pd.DataFrame:'
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Use numpy array for faster computation
    unique_ids = sorted(pd.unique(pd.concat([df["id_start"], df["id_end"]])))
    n = len(unique_ids)

    # Create id to index mapping for faster lookups
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}

    # Initialize distance matrix with infinity
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    # Vectorized initial distance filling
    start_idx = [id_to_idx[id_] for id_ in df["id_start"]]
    end_idx = [id_to_idx[id_] for id_ in df["id_end"]]
    dist_matrix[start_idx, end_idx] = df["distance"].values
    dist_matrix[end_idx, start_idx] = df["distance"].values

    # Vectorized Floyd-Warshall
    for k in range(n):
        # Broadcasting to create matrices for comparison
        dist_ik = dist_matrix[:, k : k + 1]
        dist_kj = dist_matrix[k : k + 1, :]
        new_dists = dist_ik + dist_kj

        # Update distances where new path is shorter
        dist_matrix = np.minimum(dist_matrix, new_dists)

    return pd.DataFrame(dist_matrix, index=unique_ids, columns=unique_ids)


def unroll_distance_matrix(df) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Creating mesh grid for all combinations
    ids = df.index.values
    id_end, id_start = np.meshgrid(ids, ids)  # Swapped order here

    # Flatten arrays and get corresponding distances
    id_start_flat = id_start.flatten()
    id_end_flat = id_end.flatten()
    distances = df.values.flatten()

    # Filter out diagonal elements
    mask = id_start_flat != id_end_flat

    # Create DataFrame directly from filtered arrays
    result = pd.DataFrame(
        {
            "id_start": id_start_flat[mask],
            "id_end": id_end_flat[mask],
            "distance": distances[mask],
        }
    )

    # Sort by id_start and id_end to ensure consistent ordering
    result = result.sort_values(["id_start", "id_end"]).reset_index(drop=True)

    return result


def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the given reference ID
    ref_avg_distance = df[df["id_start"] == reference_id]["distance"].mean()
    print(f"Reference ID {reference_id} - Average Distance: {ref_avg_distance}")

    # Define the ±10% threshold range
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1

    # Filter rows for the reference ID and within the threshold bounds
    relevant_rows = df[df["id_start"] == reference_id]
    matching_rows = relevant_rows[
        (relevant_rows["distance"] >= lower_bound)
        & (relevant_rows["distance"] <= upper_bound)
    ]

    # Sort the matching rows by distance in ascending order
    sorted_matching_rows = matching_rows[["id_end", "distance"]].sort_values(
        by="distance"
    )

    # Reset index and return the sorted DataFrame
    return sorted_matching_rows.reset_index(
        drop=True
    )  # returning a pd.DataFrame() instead of a sorted List[] to match the output signature


def calculate_toll_rate(df) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rates = {"moto": 0.8, "car": 1.2, "rv": 1.5, "bus": 2.2, "truck": 3.6}

    # Vectorized multiplication for all vehicle types at once
    result = df.assign(
        **{vehicle: df["distance"] * rate for vehicle, rate in rates.items()}
    )  # chain '.drop(columns=['distance'])' to this assign method to have 'distance' column removed from display
    # Not dropping the 'distance' column since it's required for inclusion in the next function in the chain. Would rather have than not have.
    return result


def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Pre-define constants
    time_intervals = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8),
    ]
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    vehicle_types = ["moto", "car", "rv", "bus", "truck"]

    # Pre-calculate the number of rows needed
    n_weekday_entries = len(df) * len(days_of_week) * len(time_intervals)
    n_weekend_entries = len(df)
    total_entries = n_weekday_entries + n_weekend_entries

    # Pre-allocate arrays for better performance
    expanded_data = {
        "id_start": np.zeros(total_entries, dtype=int),
        "id_end": np.zeros(total_entries, dtype=int),
        "distance": np.zeros(total_entries),
        "start_day": [""] * total_entries,
        "start_time": [None] * total_entries,
        "end_day": [""] * total_entries,
        "end_time": [None] * total_entries,
    }
    for vehicle in vehicle_types:
        expanded_data[vehicle] = np.zeros(total_entries)

    # Fill data using array operations where possible
    idx = 0
    for _, row in df.iterrows():
        base_tolls = {vehicle: row[vehicle] for vehicle in vehicle_types}

        # Weekday entries
        for day in days_of_week:
            for start_time, end_time, factor in time_intervals:
                for key in ["id_start", "id_end", "distance"]:
                    expanded_data[key][idx] = row[key]
                expanded_data["start_day"][idx] = day
                expanded_data["start_time"][idx] = start_time
                expanded_data["end_day"][idx] = day
                expanded_data["end_time"][idx] = end_time

                # Calculate tolls
                for vehicle in vehicle_types:
                    expanded_data[vehicle][idx] = round(base_tolls[vehicle] * factor, 2)
                idx += 1

        # Weekend entry
        for key in ["id_start", "id_end", "distance"]:
            expanded_data[key][idx] = row[key]
        expanded_data["start_day"][idx] = "Saturday"
        expanded_data["start_time"][idx] = time(0, 0, 0)
        expanded_data["end_day"][idx] = "Sunday"
        expanded_data["end_time"][idx] = time(23, 59, 59)

        # Calculate weekend tolls
        for vehicle in vehicle_types:
            expanded_data[vehicle][idx] = round(base_tolls[vehicle] * 0.7, 2)
        idx += 1

    return pd.DataFrame(expanded_data)
