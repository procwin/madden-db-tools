"""
Python tools and utilities
"""

import numpy as np
import pandas as pd


def coalesce(data, x, y, impute=np.nan):
    '''
    perform sql-like coalesce for two df columns
    '''

    d = data.copy()
    out = list(i if not i is None and not pd.isnull(i) else 
               (j if not j is None and not pd.isnull(j) else impute) for i,j in zip(d[x],d[y]))
    return out


def to_lower(data):
    '''
    convert df columns to lowercase
    '''

    d = data.copy()
    d.columns = [c.lower() for c in d.columns]

    return d


def to_numeric(data, columns=None, dtype=None, errors='ignore'):
    '''
    convert df columns to numeric if possible and attempt dtype downcast
    '''

    d = data.copy()

    if not columns:
        columns = d.columns
        d[columns] = d[columns].apply(pd.to_numeric, downcast=dtype, errors=errors)

    return d


def format_data(data):
    '''
    wrapper function for column and dtype conversion
    '''

    d = data.copy()
    d = to_lower(d)
    d = to_numeric(d, columns=None, dtype=None, errors='ignore')

    return d


def is_unique(data, cols, print_dups=True, return_dups=False):
    '''
    check if one or more columns are unique in data frame
    '''

    d = data.copy()
    if not isinstance(cols, list):
        cols = [cols]

    counts = d.groupby(cols).size().reset_index().rename(columns={0:'cnt'})
    is_unq = (sum(counts['cnt']>1) == 0)
    if not is_unq:
        bad = counts.loc[counts['cnt']>1]
        if print_dups:
            print(f"Duplicate {'/'.join([c.upper() for c in cols])}:\n{bad}\n")
        if return_dups:
            return bad
        return False
    return True
