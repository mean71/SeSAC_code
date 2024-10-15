from collections import Counter
import math

def n_grams(sequence, n):
    """Generates n-grams from a sequence of words."""
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def count_n_grams(sequence, n):
    """Counts the n-grams in a sequence."""
    return Counter(n_grams(sequence, n))

def modified_precision(candidate, reference, n):
    """Calculates modified precision for a specific n-gram."""
    candidate_ngrams = count_n_grams(candidate, n)
    reference_ngrams = count_n_grams(reference, n)
    
    # Count overlapping n-grams
    overlap = candidate_ngrams & reference_ngrams  # Intersection: counts min occurrences
    overlap_count = sum(overlap.values())
    
    # Total candidate n-grams
    total_count = sum(candidate_ngrams.values())
    
    # Avoid division by zero
    if total_count == 0:
        return 0
    
    return overlap_count / total_count

def brevity_penalty(candidate, reference):
    """Calculates the brevity penalty for the candidate translation."""
    candidate_len = len(candidate)
    reference_len = len(reference)
    
    if candidate_len > reference_len:
        return 1
    elif candidate_len == 0:  # Avoid division by zero
        return 0
    else:
        return math.exp(1 - reference_len / candidate_len)

def bleu(candidate, reference, max_n=4):
    """Calculates the BLEU score for a candidate against a reference."""
    # Calculate precisions for n=1 to n=4
    precisions = []
    for n in range(1, max_n+1):
        precisions.append(modified_precision(candidate, reference, n))
    
    # Calculate the geometric mean of the precisions (or smooth to avoid zero)
    if all(p == 0 for p in precisions):
        bleu_score = 0
    else:
        # Geometric mean
        bleu_score = math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n)
    
    # Apply brevity penalty
    bp = brevity_penalty(candidate, reference)
    
    return bleu_score * bp

if __name__ == '__main__':
    # Example usage
    reference = "this is a great test".split()
    candidate = "this was a good test".split()

    bleu = bleu(candidate, reference)
    print(f"BLEU score: {bleu:.4f}")
