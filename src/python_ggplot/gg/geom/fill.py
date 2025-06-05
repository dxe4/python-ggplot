from typing import Any, Generator, List, Tuple

import pandas as pd


def enumerate_groups(
    df: pd.DataFrame, columns: List[str]
) -> Generator[Tuple[Tuple[Any, ...], pd.DataFrame], None, None]:
    grouped = df.groupby(columns, sort=True)  # type: ignore
    sorted_keys = sorted(grouped.groups.keys(), reverse=True)  # type: ignore
    for keys in sorted_keys:  # type: ignore
        sub_df = grouped.get_group(keys)  # type: ignore
        yield keys, sub_df
