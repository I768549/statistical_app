import math

def read_distribution(file_name):
    try:
        with open(file_name, 'r') as data_file:
            data = data_file.read()
        tokens = data.replace(',', ' ').split()
        raw_dist_array = [float(token) for token in tokens]
        return raw_dist_array
    except Exception as e:
        a = ValueError("Упс :)")
        raise a

def midpoint_intervals_forming(raw_dist_array, bins = 0):
    array_lenght = len(raw_dist_array)
    maximum = max(raw_dist_array)
    minimum = min(raw_dist_array)
    if bins == 0:
        if array_lenght < 100 and int((array_lenght)**(1/2)) % 2 == 0:
            bins_amount = int((array_lenght)**(0.5)) - 1
        elif array_lenght < 100 and int((array_lenght)**(1/2)) % 2 == 1:
            bins_amount = int((array_lenght)**(0.5))
        elif array_lenght >= 100 and int((array_lenght)**(1/3)) % 2 == 0:
            bins_amount = int((array_lenght)**(1/3)) - 1
        elif array_lenght >= 100 and int((array_lenght)**(1/3)) % 2 == 1:
            bins_amount = int((array_lenght) ** (1/3))
    else:
        bins_amount = bins
    delta_h = (maximum - minimum )/bins_amount
    intervals_array = [minimum+(i+0.5)*delta_h for i in range(bins_amount)]
    return {"intervals_array": intervals_array, "delta_h": delta_h, "bins_amount": bins_amount}

def frequencies(raw_dist_array, delta_h, bins_amount):
    minimum = min(raw_dist_array)
    frequencies_array = [0] * bins_amount  # Ensure correct length

    for value in raw_dist_array:
        bin_index = min(int((value - minimum) / delta_h), bins_amount - 1)  # Prevent overflow
        frequencies_array[bin_index] += 1

    return frequencies_array

def relative_frequencies(frequencies_array, raw_dist_array_length):
    return [freq / raw_dist_array_length for freq in frequencies_array]

def pdf_from_histogram(frequencies_array, delta_h, raw_dist_array_length):
    pdf_values = [(i/raw_dist_array_length)/delta_h for i in frequencies_array]
    return pdf_values

def cumulative_sum(array):
    cum_coef = 0
    cumulative_array = []
    for i in array:
        cum_coef+=i
        cumulative_array.append(cum_coef)
    return cumulative_array

def ecdf(raw_dist_data):
    sorted_dist_data = sorted(raw_dist_data)
    length = len(sorted_dist_data)
    y_axis = [(i+1)/length for i in range(length)]
    return sorted_dist_data, y_axis

def arithmetic_mean(raw_dist_data):
    sum = 0
    for element in raw_dist_data:
        sum += element
    return sum/len(raw_dist_data)

def sample_median(raw_dist_data):
    sorted_dist_data = sorted(raw_dist_data)
    length = len(sorted_dist_data)
    if length % 2 == 1: #odd
        med = sorted_dist_data[int(length/2)]
    else: #even
        med = (sorted_dist_data[int((length-1)/2)] + sorted_dist_data[int(length/2)])/2
    return med

def trimmed_mean(raw_dist_data, a = 0.1): # trim a% from both sides, then find arithmetic mean on what is left
    sorted_dist_data = sorted(raw_dist_data)
    length = len(sorted_dist_data)
    trimmed_sum = 0
    k = int(a*length)
    t = k
    while t < length-k:
        trimmed_sum += sorted_dist_data[t]
        t+=1
    return trimmed_sum/(length - 2*k)

def walsh_median(raw_dist_data, intervals, delta_h):
    if not intervals or not raw_dist_data:
        return 0  # Handle edge case

    bins_amount = len(intervals)
    minimum = intervals[0] - 0.5 * delta_h  # Infer minimum from first midpoint
    bin_edges = [minimum + i * delta_h for i in range(bins_amount + 1)]

    # Assign values to bins
    values_for_bins = [[] for _ in range(bins_amount)]
    for value in raw_dist_data:
        for i in range(bins_amount):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                values_for_bins[i].append(value)
                break
        else:
            # Include maximum value in the last bin
            if value == bin_edges[-1]:
                values_for_bins[-1].append(value)

    # Calculate means for non-empty bins
    mean_values_for_bins = []
    for bin_values in values_for_bins:
        if bin_values:  # Only include non-empty bins
            mean_values_for_bins.append(sum(bin_values) / len(bin_values))

    # Return average of bin means, or 0 if all bins are empty
    return sum(mean_values_for_bins) / len(mean_values_for_bins) if mean_values_for_bins else 0

def unbiased_sample_variance(raw_dist_data, mean):
    variance_sum = 0
    for element in raw_dist_data:
        variance_sum += (element-mean)**2
    return variance_sum/(len(raw_dist_data) - 1)

def biased_sample_variance(raw_dist_data, mean):
    variance_sum = 0
    for element in raw_dist_data:
        variance_sum += element**2 - mean**2
    return variance_sum/len(raw_dist_data)

def biased_asymmetry(raw_dist_data, mean, biased_sample_standart_deviation):
    asymmetry_sum = 0
    for element in raw_dist_data:
        asymmetry_sum += (element - mean) ** 3
    return asymmetry_sum/(len(raw_dist_data)* biased_sample_standart_deviation**3)

def unbiased_asymmetry(length, biased_asymmetry_value):
    unbiased_asymmetry_var = biased_asymmetry_value * (((length*(length-1))**(1/2))/(length - 2))
    return unbiased_asymmetry_var

def biased_kurtosis(raw_dist_data, mean, biased_sample_standart_deviation):
    kurtosis_sum = 0
    for element in raw_dist_data:
        kurtosis_sum += (element - mean) ** 4
    return kurtosis_sum / (len(raw_dist_data) * biased_sample_standart_deviation**4)

def unbiased_kurtosis(length, biased_kurtosis_value):
    unbiased_kurtosis_sum_var = ((length**2 - 1)/((length-2)*(length-3)))*((biased_kurtosis_value - 3) + (6/(length+1)))
    return unbiased_kurtosis_sum_var

def counter_kurtosis(unbiased_kurtosis_value):
    chita = 1/((abs(unbiased_kurtosis_value))**(1/2)) #X
    return chita

def pirson_coeff(unbiased_sample_standart_deviation, mean):
    if mean != 0:
        coeff = unbiased_sample_standart_deviation/mean
    else:
        coeff = 0
    return coeff

def shift_data(raw_dist_array, shift_coeff):
    shifted_array = [element+shift_coeff for element in raw_dist_array]
    return shifted_array

def logarithmize_data(positive_dist_array):
    data_list = list(positive_dist_array)

    for value in data_list:
        if value <= 0:
            raise ValueError("All values must be positive for logarithm transformation")

    return [math.log(value) for value in data_list]

def trim_data(raw_dist_data, a = 0.25):
    sorted_dist_data = sorted(raw_dist_data)
    length = len(sorted_dist_data)
    k = int(a * length)
    trimmed_array = sorted_dist_data[k:length - k]
    return trimmed_array

def del_anomaly_data_Z_score(raw_dist_data, mean, unbiased_standart_deviation, lower_z = -1.5, upper_z = 1.5):
    if unbiased_standart_deviation == 0:
        return raw_dist_data
    no_anomaly_data = []
    for element in raw_dist_data:
        z = (element - mean)/unbiased_standart_deviation
        if z > lower_z and z < upper_z:
            no_anomaly_data.append(element)
        elif element == mean:
             no_anomaly_data.append(element)

    return no_anomaly_data

def anomaly_deletion_by_unbiased_kurtosis(raw_dist_data, unbiased_standard_deviation, unbiased_kurtosis, unbiased_assymetry, x_mean):
    t_1 = 2 + 0.2*math.log10(0.04*len(raw_dist_data))
    t_2 = (19*(unbiased_kurtosis + 2)**(1/2) +1)**(1/2)
    if unbiased_assymetry < -0.2:
        a = x_mean - t_2*unbiased_standard_deviation
        b = x_mean + t_1*unbiased_standard_deviation
    elif unbiased_assymetry > 0.2:
        a = x_mean - t_1*unbiased_standard_deviation
        b = x_mean + t_2*unbiased_standard_deviation
    elif abs(unbiased_assymetry) <= 0.2:
        a = x_mean - t_1*unbiased_standard_deviation
        b = x_mean + t_1*unbiased_standard_deviation
    no_anomalies_array = []
    for element in raw_dist_data:
        if b > element > a:
            no_anomalies_array.append(element)
    return no_anomalies_array

def standartise_data(raw_dist_data):
    mean = arithmetic_mean(raw_dist_data)
    unbiased_standart_deviation = (unbiased_sample_variance(raw_dist_data, mean))**(1/2)
    if unbiased_standart_deviation == 0:
        return raw_dist_data
    return [(element - mean)/unbiased_standart_deviation for element in raw_dist_data]

def standard_error_of_mean(unbiased_standard_deviation, sample_size):
    if sample_size <= 1:
        raise ValueError("Sample size is wrong!")
    return unbiased_standard_deviation/sample_size**(1/2)


def confidence_interval_mean(mean, std_error, sample_size, t_crit = 2):
    if sample_size <= 1:
        raise ValueError("Sample size is wrong!")
    degrees_freedom = sample_size - 1
    margin_error = t_crit * std_error
    return mean - margin_error, mean + margin_error

def standard_error_of_variance(unbiased_variance, sample_size):
    variance_term = (2 * (unbiased_variance ** 2)) / (sample_size - 1)
    return variance_term**(1/2)

def confidence_interval_variance(unbiased_variance, sample_size):
    if sample_size <= 1:
        return (0.0, float('inf'))
    if unbiased_variance < 0:
        raise ValueError("Variance cannot be negative.")
    
    df = sample_size - 1
    # For α = 0.025 (right tail)
    if df == 1:
        chi2_upper = 5.024
    elif df <= 5:
        # Approximate for small df values
        chi2_upper = 4.0 + df
    elif df <= 30:
        # Approximate for medium df values
        chi2_upper = df + 1.96 * math.sqrt(2 * df)
    else:
        # Approximate for large df values
        chi2_upper = df + 2 * math.sqrt(df)
    
    if df == 1:
        chi2_lower = 0.001
    elif df <= 5:
        chi2_lower = 0.2 * df
    elif df <= 30:
        chi2_lower = df - 1.96 * math.sqrt(2 * df)
    else:
        # Approximate for large df values
        chi2_lower = df - 2 * math.sqrt(df)
    
    lower_bound = (df * unbiased_variance / chi2_upper)
    upper_bound = (df * unbiased_variance / chi2_lower)
    
    return lower_bound, upper_bound

def standard_error_of_asymmetry(sample_size):
    if sample_size <= 2:
        return float('inf')
    numerator = 6 * sample_size * (sample_size - 1)
    denominator = (sample_size - 2) * (sample_size + 1) * (sample_size + 3)
    if denominator == 0:
        return float('inf')
    return (numerator / denominator)**(1/2)

def norm_critical_value(confidence=0.95):
    # Hard-coded z-values for common confidence levels
    z_table = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576
    }
    # Only supporting 95% confidence level in this version
    if confidence != 0.95:
        raise ValueError("Only 0.95 confidence level is supported in this version")
    
    return z_table[confidence]

def confidence_interval_asymmetry(asymmetry, se_asymmetry):
    z_crit = norm_critical_value(0.95)
    margin_error = z_crit * se_asymmetry
    return asymmetry - margin_error, asymmetry + margin_error

def standard_error_of_kurtosis(sample_size):
    if sample_size <= 3:
        return float('inf')
    numerator = 24 * sample_size * (sample_size - 1) ** 2
    denominator = (sample_size - 3) * (sample_size - 2) * (sample_size + 3) * (sample_size + 5)
    if denominator == 0: return float('inf')
    return (numerator / denominator)**(1/2)

def confidence_interval_kurtosis(excess_kurtosis, se_kurtosis):
    z_crit = norm_critical_value(0.95)
    margin_error = z_crit * se_kurtosis
    return excess_kurtosis - margin_error, excess_kurtosis + margin_error

def prediction_interval(mean, unbiased_std, sample_size, t_crit = 2):
    if sample_size <= 1:
        return (float('-inf'), float('inf')) # Undefined

    # Formula: mean ± t_crit * std * sqrt(1 + 1/n)
    margin = t_crit * unbiased_std * (1 + 1 / sample_size)**(1/2)
    return mean - margin, mean + margin

#MAD
def median_absolute_deviation(raw_dist_data):
    #MED
    median = sample_median(raw_dist_data)
    median_subst_array = [abs(element - median) for element in raw_dist_data]
    median_absolute_deviation = 1.483 * sample_median(median_subst_array)
    return median_absolute_deviation

def calculate_all_intervals(raw_dist_data):
    if not raw_dist_data:
        raise ValueError("Input data cannot be empty for interval calculations.")

    sample_size = len(raw_dist_data)
    if sample_size <= 1:
        raise ValueError("Need at least 2 data points for most interval calculations.")

    # --- Calculate Basic Statistics ---
    mean = arithmetic_mean(raw_dist_data)
    unbiased_var = unbiased_sample_variance(raw_dist_data, mean)
    unbiased_std = unbiased_var**(1/2)

    biased_var = biased_sample_variance(raw_dist_data, mean)
    biased_std = biased_var**(1/2)

    biased_asym = biased_asymmetry(raw_dist_data, mean, biased_std)
    unbiased_asym = unbiased_asymmetry(sample_size, biased_asym)

    biased_kurt = biased_kurtosis(raw_dist_data, mean, biased_std)
    unbiased_exc_kurt = unbiased_kurtosis(sample_size, biased_kurt)

    # --- Calculate Standard Errors ---
    se_mean = standard_error_of_mean(unbiased_std, sample_size)
    se_variance = standard_error_of_variance(unbiased_var, sample_size)
    se_asymmetry = standard_error_of_asymmetry(sample_size)
    se_kurtosis = standard_error_of_kurtosis(sample_size)

    # --- Calculate Confidence Intervals ---
    ci_mean = confidence_interval_mean(mean, se_mean, sample_size)
    ci_variance = confidence_interval_variance(unbiased_var, sample_size)
    ci_std = ((max(0, ci_variance[0]))**(1/2), (max(0, ci_variance[1]))**(1/2))
    ci_asymmetry = confidence_interval_asymmetry(unbiased_asym, se_asymmetry)
    ci_kurtosis = confidence_interval_kurtosis(unbiased_exc_kurt, se_kurtosis)

    # --- Calculate Prediction Interval ---
    pred_interval = prediction_interval(mean, unbiased_std, sample_size) if sample_size > 1 else (float('nan'), float('nan'))

    results = {
        "Standard Errors": {
            "SE Mean": se_mean,
            "SE Variance": se_variance,
            "SE Skewness": se_asymmetry,
            "SE Excess Kurtosis": se_kurtosis
        },
        "Confidence Intervals": {
            "CI Mean": ci_mean,
            "CI Variance": ci_variance,
            "CI Std Dev": ci_std,
            "CI Skewness": ci_asymmetry,
            "CI Excess Kurtosis": ci_kurtosis
        },
        "Prediction Intervals": {
            "PI Single Observation": pred_interval
        }
    }

    return results