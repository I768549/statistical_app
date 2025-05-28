import math
import numpy as np
from main_functions import *

def kolmogorov_refined_test(theoretical_cdf, x_values, empirical_cdf):
    n = len(x_values)

    # D⁺ and D⁻
    D_plus = np.max(empirical_cdf - theoretical_cdf(x_values))
    D_minus = np.max(theoretical_cdf(x_values) - empirical_cdf)
    D_n = max(D_plus, D_minus)

    z = np.sqrt(n) * D_n

    # Refined K(z)
    K = 1.0
    k_max = 50
    for k in range(1, k_max + 1):
        f1 = k**2 - 0.5 * (1 - (-1)**k)
        f2 = 5 * k**2 + 22 - 7.5 * (1 - (-1)**k)

        base = (-1)**k * np.exp(-2 * k**2 * z**2)

        # Основна корекція (внутрішнє множення в першій дужці)
        term_main = (
            1
            - (2 * k**2 * z**2) / (3 * n)
            - (1 / (18 * n)) * ((f1 - 4 * (f1 + 3)) * k**2 * z**2 + 8 * k**4 * z**4)
        )

        # Друга корекція — виноска з правої частини формули
        term_tail = (
            (k**2 * z) / (27 * np.sqrt(n**3))
            * (
                (f2**2) / 5
                - (4 * (f2 + 45) * k**2 * z**2) / 15
                + 8 * k**4 * z**4
            )
        )

        K += 2 * base * (term_main + term_tail)

    p_value = 1 - K

    return {
        'D_n': D_n,
        'z': z,
        'p_value': p_value
    }

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

def mid_rank_diff_criteria(sample1, sample2):
    first = np.array(sample1)
    second = np.array(sample2)
    size_one = len(first)
    size_two = len(second)
    N = size_one + size_two
    ranks = rank_data(first, second)

    rank_sum1 = sum(rank for rank, group in ranks if group == 1)
    rank_sum2 = sum(rank for rank, group in ranks if group == 2)

    rx_mean = rank_sum1/size_one
    ry_mean = rank_sum2/size_two

    denominator = N * math.sqrt((N+1)/(12*size_one*size_two))

    v_stat = (rx_mean - ry_mean)/denominator
    return v_stat

def abbe_independence_criteria(sample):
    results = []
    results.append("Abbe criteria (Independence check) for the entered distribution:")
    first = np.array(sample)
    N = len(first)
    
    if N < 2:
        raise ValueError("Sample size must be at least 2 for Abbe criterion")

    mean_first = arithmetic_mean(first)
    S2 = unbiased_sample_variance(first, mean_first)

    squared_diff_sum = np.sum((first[1:] - first[:-1])**2)
    d2 = squared_diff_sum / (N - 1)
    q = d2 / (2 * S2)
    
    E_q = 1 
    D_q = (N - 2) / (N^2 - 1)
    
    U = (q - 1) * np.sqrt((N**2 - 1) / (N - 2))
    return U
