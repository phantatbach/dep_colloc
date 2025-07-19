import os
from collections import Counter

def count_lemma(corpus_path, file_ext=None, mode='lemma_pos'):
    """
    Count frequencies according to `mode`:
      - 'lemma/pos_init': lemma + "\t" + POS‐initial
      - 'lemma/deprel':     lemma + "\t" + deprel
      - 'lemma/pos':        lemma + "\t" + full POS
    """
    lemma_counts = Counter()

    for root, _, files in os.walk(corpus_path):
        for fname in files:
            if file_ext and not fname.endswith(file_ext):
                continue
            path = os.path.join(root, fname)
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('<s') or line.startswith('</s>'):
                        continue
                    
                    # Split the lines into different parts
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    # Default: wordform, lemma, pos, id, head, deprel
                    # RSC: wordform, pos, lemma, id, head, deprel
                    lemma, pos, deprel = parts[1], parts[2], parts[5]
                    pos_initial = pos[0]

                    if mode == 'lemma_pos_init':
                        key = f"{lemma}/{pos_initial}"
                    elif mode == 'lemma_deprel':
                        key = f"{lemma}/{deprel}"
                    elif mode == 'lemma_pos':
                        key = f"{lemma}/{pos}"
                    else:
                        raise ValueError(f'Mode must be one of "lemma_pos_init", "lemma_deprel", or "lemma_pos", got {mode}')

                    # Count
                    lemma_counts[key] += 1

    return dict(lemma_counts)


def save_freqs(freq_dict, out_folder, mode='lemma_pos'):
    """
    Write out `<mode>_freq.txt` into out_folder.
    Each line: key<TAB>frequency
    """
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{mode}_freq.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        for key, freq in sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True):
            out.write(f"{key}\t{freq}\n")
    return out_path


def gen_lemma_freq(corpus_path, out_folder, file_ext=None, mode='lemma_pos'):
    """
    Complete pipeline: count then save.
    Returns the path to the file written.
    """
    freqs = count_lemma(corpus_path, file_ext, mode)
    return save_freqs(freqs, out_folder, mode)