from collections import defaultdict
import re
# --- Build syntactic graph from a sentence ---

pattern = re.compile(r'([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)')
# def build_graph(tokens, pattern):
#     """
#     Given a list of conllu‐style token lines and a compiled regex pattern,
#     returns three things for the sentence:
#       1) id2lemma_pos:   { token_id: "lemma/POS" }
#       2) graph:          adjacency list { id: [neighbor_id, ...], ... }
#       3) id2deprel:      { (id1, id2): deprel, (id2, id1): deprel }
#     """
#     id2lemma_pos = {}
#     graph        = defaultdict(list)
#     id2deprel    = {}

#     for tok in tokens:
#         m = pattern.match(tok)
#         if not m:
#             continue
#         _, lemma, pos, idx, head, deprel = m.groups()
#         id2lemma_pos[idx] = f"{lemma}/{pos}"

#         if head != '0':
#             # undirected adjacency
#             graph[idx].append(head)
#             graph[head].append(idx)
#             # edge labels in both directions
#             id2deprel[(idx, head)] = deprel
#             id2deprel[(head, idx)] = deprel

#     return id2lemma_pos, graph, id2deprel

def build_graph(tokens, pattern):
    """
    Given a list of conllu-style token lines and a compiled regex pattern,
    returns three things:
      1) id2lemma_pos:   {token_id: "lemma/POS"}
      2) graph:          undirected adjacency list {id: [neighbor_id, ...], ...}
      3) id2deprel:      {(a, b): deprel_from_line_of_a, ...}
    """
    id2lemma_pos = {}
    graph = defaultdict(list)
    id2deprel = {}
    row_info = {}

    # First pass: collect lemma/pos and own deprel/head
    for tok in tokens:
        m = pattern.match(tok)
        if not m:
            continue
        _, lemma, pos, idx, head, deprel = m.groups()
        id2lemma_pos[idx] = f"{lemma}/{pos}"
        row_info[idx] = {'head': head, 'deprel': deprel}

    # Second pass: build undirected graph and assign deprel labels correctly
    for idx, info in row_info.items():
        head = info['head']
        if head != '0' and head in id2lemma_pos:
            # add both directions for traversal
            graph[idx].append(head)
            graph[head].append(idx)
            # deprel from dependent->head
            id2deprel[(idx, head)] = info['deprel']
            # deprel from head->dependent using head's own rel
            id2deprel[(head, idx)] = row_info[head]['deprel']

    return id2lemma_pos, graph, id2deprel

