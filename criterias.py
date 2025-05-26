import math
import numpy as np
def rank_data(sample1, sample2):
    combined = [(x,1) for x in sample1] + [(y,2) for y in sample2]
    combined.sort(key=lambda x: x[0])
    ranks = []
    i = 0
    while i < len(combined):
        same_value = [combined[i]]
        j = i+1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            same_value.append(combined[j])
            j += 1
        avg_rank = sum(range(i+1, j+1)) / len(same_value)
        for value in same_value:
            ranks.append((avg_rank, value[1]))
        i = j
    return ranks

def mann_whitney_u(sample1, sample2):
    # getting ranks
    first = np.array(sample1)
    second = np.array(sample2)

    size_one = len(first)
    size_two = len(second)
    N = size_one + size_two

    ranks = rank_data(first, second)

    rank_sum1 = sum(rank for rank, group in ranks if group == 1)
    rank_sum2 = sum(rank for rank, group in ranks if group == 2)

    U1 = size_one*size_two + (size_one*(size_one-1))/2.0 - rank_sum1
    U2 = size_one*size_two + (size_two*(size_two-1))/2.0 - rank_sum2
    u = max(U1, U2)
    mean_u = size_one*size_two/2.0
    disp_u = size_one*size_two*(N+1)/12

    u_statistics = (u - mean_u)/np.sqrt(disp_u)

    return u_statistics

def wilcoxon_w(sample1, sample2):
    first = np.array(sample1)
    second = np.array(sample2)
    size_one = len(first)
    size_two = len(second)
    N = size_one + size_two

    ranks = rank_data(first, second)
    rank_sum1 = sum(rank for rank, group in ranks if group == 1)
    mean_w = size_one*(N+1)/2
    disp_w = size_one*size_two*(N+1)/12
    w_stat = (rank_sum1 - mean_w)/math.sqrt(disp_w)

    return w_stat

def sign_test(first, second):
    if len(first) != len(second):
        raise ValueError("Arrays must have the same length for sign test")
    differences = [x - y for x, y in zip(first, second)]
    
    non_zero_diffs = [d for d in differences if d != 0]
    n = len(non_zero_diffs)
    
    if n == 0:
        raise ValueError("Arrays must have the same length for sign test")

    s_plus = sum(1 for d in non_zero_diffs if d > 0)
    if n > 15:
        S = (2*s_plus - 1 - n) / (n**0.5)
        return S, n
    else:
        total = 0.0
        for l in range(N - S + 1):
            numerator = math.factorial(l)
            denominator = math.factorial(N) * math.factorial(N - l)
            total += numerator / denominator
        result = (1 / (2 ** N)) * total
        return result, n


"""
import numpy as np
from scipy.stats import norm

def wilcoxon_signed_rank(sample1, sample2):
    if len(sample1) != len(sample2):
        raise ValueError("Samples must be of equal length for Wilcoxon signed-rank test.")
    
    # Compute differences
    differences = [x - y for x, y in zip(sample1, sample2)]
    
    # Remove zero differences and keep track of original indices
    nonzero_diffs = [(d, i) for i, d in enumerate(differences) if d != 0]
    if not nonzero_diffs:
        return 0  # If all differences are zero, return a neutral statistic
    
    # Rank absolute differences
    abs_diffs = sorted([(abs(d), i) for d, i in nonzero_diffs])
    ranks = []
    i = 0
    while i < len(abs_diffs):
        same_value = [abs_diffs[i]]
        j = i + 1
        while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
            same_value.append(abs_diffs[j])
            j += 1
        avg_rank = sum(range(i + 1, j + 1)) / len(same_value)  # Average rank for ties
        for value in same_value:
            ranks.append((avg_rank, value[1]))
        i = j
    
    # Assign signs to ranks
    signed_ranks = []
    for rank, idx in ranks:
        original_diff = differences[nonzero_diffs[idx][1]]
        sign = 1 if original_diff > 0 else -1
        signed_ranks.append(sign * rank)
    
    # Compute W+ (sum of positive ranks) and W- (sum of negative ranks)
    W_plus = sum(r for r in signed_ranks if r > 0)
    W_minus = -sum(r for r in signed_ranks if r < 0)
    W = min(W_plus, W_minus)
    
    # Compute z-statistic for large samples
    n = len(nonzero_diffs)
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24
    z_stat = (W - mean_W) / np.sqrt(var_W)
    
    return z_stat

"""