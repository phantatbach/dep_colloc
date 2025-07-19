import pandas as pd
import numpy as np

def PPMI_colloc_df(dep_colloc_path,
                   lemma_pos_freq_path,
                   lemma_pos_deprel_freq_path=None,
                   min_count=None,
                   mode=None):
    
    # 1) load collocation counts
    df = pd.read_csv(dep_colloc_path, index_col=0)

    # 2) load row frequencies f_i
    pos_freq = {}
    with open(lemma_pos_freq_path, encoding='utf-8') as f:
        for line in f:
            key, val = line.rstrip().split('\t', 1)
            pos_freq[key] = int(val)
    row_freq = (pd.Series(pos_freq)
                .reindex(df.index) # Re-index to match the index of pos_freq with that of df
                .fillna(0)
                .astype(int))

    # 3) load column frequencies f_j
    if mode == 'lemma_pos' or mode == 'lemma_pos_init': # If mode = lemma_pos then rows == columns
        col_freq = (row_freq
                    .reindex(df.columns) # Re-index to match the index of pos_freq with that of df
                    .fillna(0)
                    .astype(int))
    
    elif mode == 'lemma_deprel': # If mode == lemma_deprel then the columns would be deprels while rows are lemmas
        if lemma_pos_deprel_freq_path:
            deprel_freq = {}
            with open(lemma_pos_deprel_freq_path, encoding='utf-8') as f:
                for line in f:
                    key, val = line.rstrip().split('\t', 1)
                    deprel_freq[key] = int(val)
            col_freq = (pd.Series(deprel_freq) 
                        .reindex(df.columns) # Re-index to match the index of pos_freq with that of df
                        .fillna(0)
                        .astype(int))

        else:
            raise ValueError("lemma_pos_deprel_freq_path is required for mode 'lemma_deprel'")
    
    else:
        raise ValueError("Invalid mode: must be 'lemma_pos', 'lemma_pos_init', or 'lemma_deprel'")

    # 4) total N
    N = df.values.sum()

    # 5) prepare empty PPMI frame
    ppmi = pd.DataFrame(
        np.zeros(df.shape, dtype=float),
        index=df.index,
        columns=df.columns
    )

    # 6) compute PPMI
    for i in df.index:
        f_i = row_freq.at[i]
        if f_i == 0:
            continue
        for j in df.columns:
            n_ij = df.at[i, j]
            if min_count is not None and n_ij < min_count:
                continue
            f_j = col_freq.at[j]
            if n_ij > 0 and f_j > 0:
                pmi = np.log2((n_ij * N) / (f_i * f_j))
                ppmi.at[i, j] = max(pmi, 0.0)

    return ppmi