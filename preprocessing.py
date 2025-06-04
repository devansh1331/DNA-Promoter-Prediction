import numpy as np

def create_sequence_features(sequence):
    """
    Extract meaningful features from DNA sequences.
    Must match exactly what was used during training.
    """
    seq = sequence.replace('\t', '').strip().lower()
    
    features = {}

    # Basic nucleotide composition
    for nucleotide in ['a', 't', 'g', 'c']:
        features[f'{nucleotide}_count'] = seq.count(nucleotide)
        features[f'{nucleotide}_freq'] = seq.count(nucleotide) / len(seq) if len(seq) > 0 else 0

    # Dinucleotide features
    dinucleotides = ['aa', 'at', 'ag', 'ac', 'ta', 'tt', 'tg', 'tc',
                     'ga', 'gt', 'gg', 'gc', 'ca', 'ct', 'cg', 'cc']
    
    for dinuc in dinucleotides:
        count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == dinuc)
        features[f'{dinuc}_count'] = count
        features[f'{dinuc}_freq'] = count / (len(seq) - 1) if len(seq) > 1 else 0

    # GC content
    gc_count = seq.count('g') + seq.count('c')
    features['gc_content'] = gc_count / len(seq) if len(seq) > 0 else 0

    # CpG island detection
    cg_count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == 'cg')
    features['cg_count'] = cg_count
    features['cg_density'] = cg_count / (len(seq) - 1) if len(seq) > 1 else 0
    
    return features

def preprocess_sequence_for_prediction(sequence):
    """
    Preprocess a single DNA sequence for model prediction.
    """
    features = create_sequence_features(sequence)

    # Feature order - must match exactly with model training features
    feature_order = [
        'a_count', 'a_freq', 't_count', 't_freq', 'g_count', 'g_freq', 'c_count', 'c_freq',
        'aa_count', 'aa_freq', 'at_count', 'at_freq', 'ag_count', 'ag_freq', 'ac_count', 'ac_freq',
        'ta_count', 'ta_freq', 'tt_count', 'tt_freq', 'tg_count', 'tg_freq', 'tc_count', 'tc_freq',
        'ga_count', 'ga_freq', 'gt_count', 'gt_freq', 'gg_count', 'gg_freq', 'gc_count', 'gc_freq',
        'ca_count', 'ca_freq', 'ct_count', 'ct_freq', 'cg_count', 'cg_freq', 'cc_count', 'cc_freq',
        'gc_content', 'cg_count', 'cg_density'  # Removed 'sequence_length' to match model training
    ]

    # Ensure correct feature count
    feature_vector = [features[feature] for feature in feature_order]

    return np.array(feature_vector).reshape(1, -1)

def validate_dna_sequence(sequence):
    """
    Validate if the input is a proper DNA sequence.
    """
    sequence = sequence.upper().strip()
    valid_chars = set('ATGCN')  # N represents unknown nucleotide

    if not sequence:
        return False, "Empty sequence"

    if len(sequence) < 10:
        return False, "Sequence too short (minimum 10 nucleotides)"

    invalid_chars = set(sequence) - valid_chars
    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(invalid_chars)}"

    return True, "Valid sequence"
