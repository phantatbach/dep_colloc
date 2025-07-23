from collections import defaultdict
import re
# --- Build syntactic graph from a sentence ---

pattern = re.compile(r'([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)')

def build_graph(tokens, pattern):
    """
    Given a list of conllu tokens and a regex pattern, build a dependency graph,
    a mapping of id to lemma/pos, and a mapping of edge to deprel.

    Returns a tuple of (id2lemma_pos, graph, id2deprel).
    """
    
    id2lemma_pos = {}
    graph        = defaultdict(list)
    id2deprel    = {}

    for tok in tokens:
        m = pattern.match(tok)
        if not m:
            continue
        # You don't need the wordform to create a dependency graph
        _, lemma, pos, idx, head, deprel = m.groups()
        # Create a dictionary of {id: lemma/pos}
        id2lemma_pos[idx] = f'{lemma}/{pos}'
        # If the current word is not the root
        if head != '0':
            # Create an un-directional graph of with 2 dictionary entries {children: [parents]} and {parents: [children]}
            graph[idx].append(head)
            graph[head].append(idx)

            # Create a un-directional graph with 2 dictionary entries {edge: deprel}
            # chi: child-ward, pa: parent-ward
            id2deprel[(idx, head)] = f'pa_{deprel}'
            id2deprel[(head, idx)] = f'chi_{deprel}'
    return id2lemma_pos, graph, id2deprel

# def build_graph(tokens, pattern):
#     """
#     Given a list of conllu-style token lines and a compiled regex pattern,
#     returns three things:
#       1) id2lemma_pos:   {token_id: "lemma/POS"}
#       2) graph:          undirected adjacency list {id: [neighbor_id, ...], ...}
#       3) id2deprel:      {(a, b): deprel_from_line_of_a, ...}
#     """
#     id2lemma_pos = {}
#     graph = defaultdict(list)
#     id2deprel = {}
#     row_info = {}

#     # First pass: collect lemma/pos and own deprel/head
#     for tok in tokens:
#         m = pattern.match(tok)
#         if not m:
#             continue
#         _, lemma, pos, idx, head, deprel = m.groups()
#         id2lemma_pos[idx] = f"{lemma}/{pos}"
#         row_info[idx] = {'head': head, 'deprel': deprel}

#     # Second pass: build undirected graph and assign deprel labels correctly
#     for idx, info in row_info.items():
#         head = info['head']
#         if head != '0' and head in id2lemma_pos:
#             # add both directions for traversal
#             graph[idx].append(head)
#             graph[head].append(idx)
#             # deprel from dependent->head
#             id2deprel[(idx, head)] = info['deprel']
#             # deprel from head->dependent using head's own rel
#             id2deprel[(head, idx)] = row_info[head]['deprel']

#     return id2lemma_pos, graph, id2deprel

