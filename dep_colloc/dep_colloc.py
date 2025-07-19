import os
import re
from collections import defaultdict, Counter, deque
import pandas as pd
from dep_colloc.utils import build_graph
from tqdm import tqdm

def generate_syn_colloc_df(corpus_dir, max_depth, pattern):
    """
    Reads every file under `corpus_dir`, processes each sentence to build
    a dependency graph, and for each token (row_key = lemma/POS) performs
    a BFS up to `max_depth`, recording each context token with its correct
depe rel.

    Returns a pandas.DataFrame with:
      index = lemma/POS of target
      columns = lemma/POS of context + "/" + deprel
      values = counts
    """
    colloc = Counter()
    files = sorted([f for f in os.listdir(corpus_dir) if f.endswith('.txt')])

    for filename in tqdm(files, desc='Processing files'):
        path = os.path.join(corpus_dir, filename)
        with open(path, encoding='utf-8') as f:
            tokens = []
            for line in f:
                line = line.strip()
                if line.startswith('<s'):
                    tokens = []
                    continue
                if line.startswith('</s>'):
                    # build graph for current sentence
                    id2lemma_pos, graph, id2deprel = build_graph(tokens, pattern)
                    # BFS for each token
                    for start_id, target_lp in id2lemma_pos.items():
                        seen = {start_id}
                        queue = deque([(start_id, 0)])
                        while queue:
                            curr, depth = queue.popleft()
                            if depth >= max_depth:
                                continue
                            for nb in graph.get(curr, []):
                                if nb in seen:
                                    continue
                                seen.add(nb)
                                # get correct rel, whichever direction
                                rel = id2deprel.get((nb, curr)) or id2deprel.get((curr, nb))
                                ctx_lp = id2lemma_pos[nb].split('/')[0]
                                col_key = f"{ctx_lp}/{rel}"
                                colloc[(target_lp, col_key)] += 1
                                queue.append((nb, depth + 1))
                    tokens = []
                    continue
                if line:
                    tokens.append(line)

    # pivot into DataFrame
    data = defaultdict(dict)
    for (row_key, col_key), cnt in colloc.items():
        data[row_key][col_key] = cnt

    df = pd.DataFrame.from_dict(data, orient='index')
    df.fillna(0, inplace=True)
    df.index.name = 'lemma_pos'
    return df


def generate_path_colloc_df(corpus_dir, max_depth, pattern):
    """
    Walks every file under `corpus_dir`, reads sentences in this form:
        <s id=...>
        wordform  lemma  pos  id  head  deprel
        ...
        </s>
    Builds a dependency graph for each sentence, then for each token
    (row_key = lemma/pos) does a BFS out to `max_depth`, and for every
    edge traversed records a (lemma/pos, neighborLemma/neighborPos) pair.
    
    Returns
    -------
    pandas.DataFrame
        index = all lemma/pos seen
        columns = all lemma/pos seen
        values = counts of how often they co‐occur within depth
    """
    colloc_counts = defaultdict(int)
    all_vocab = set()

    files = sorted([f for f in os.listdir(corpus_dir) if f.endswith(".txt")])
    for filename in tqdm(files, desc="Processing files"):
        with open(os.path.join(corpus_dir, filename), encoding='utf-8') as f:
            lines = f.readlines()

        sentence = []
        for line in lines:
            line = line.strip()
            
            if line.startswith("<s"):
                sentence = []

            elif line.startswith("</s>"):
                id2lemma_pos, graph, id2deprel = build_graph(sentence, pattern)
                for idx, token in id2lemma_pos.items():
                    visited = set()
                    queue = deque([(idx, 0)])
                    while queue:
                        current, depth = queue.popleft()
                        if current in visited or depth > max_depth:
                            continue
                        visited.add(current)

                        if depth > 0 and current in id2lemma_pos:
                            neigh = id2lemma_pos[current]
                            # ensure ordering so (a,b) == (b,a)
                            key = tuple(sorted((token, neigh)))
                            colloc_counts[key] += 1

                        for nbr in graph.get(current, []):
                            queue.append((nbr, depth + 1))

                all_vocab.update(id2lemma_pos.values())

            elif line:
                sentence.append(line)

    # build vocab list
    vocab = sorted(all_vocab)
    # initialize empty DataFrame
    df = pd.DataFrame(0, index=vocab, columns=vocab, dtype=int)

    # fill counts (symmetric)
    for (tok1, tok2), count in colloc_counts.items():
        df.at[tok1, tok2] = count
        df.at[tok2, tok1] = count

    return df