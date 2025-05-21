import random
import math
from scipy.special import gamma
from scipy.optimize import fsolve

def generate_exp_theoretical_dist(size, lam = 1):
    return [(math.log(1 - random.random())/-lam) for _ in range(size)]

def generate_uniform_theoretical_dist(size, a, b):
    return [a + (b-a)*random.random() for _ in range(size)]

def generate_weibull_theoretical_dist(size, alpha, beta):
    return [alpha *((-math.log(1 - random.random()))**(1/beta)) for _ in range(size)]

def generate_normal_box_muller_distribution(size, mean, std):
    #standard
    #standard_1 = [(math.sqrt(-2*math.log(random.random()))) * math.cos(2*math.pi*random.random()) for _ in range(size)]
    sample_1 = [mean + std * ((math.sqrt(-2*math.log(random.random()))) * math.cos(2*math.pi*random.random())) for _ in range(size)]
    #standard_2
    #standard_2 = [(math.sqrt(-2*math.log(random.random()))) * math.sin(2*math.pi*random.random()) for _ in range(size)]
    #sample_2 = [mean + std * ((math.sqrt(-2*math.log(random.random()))) * math.sin(2*math.pi*random.random())) for _ in range(size)]
    return sample_1
def generate_log_normally_distribution(x):
    return [math.exp(element) for element in x]

def generate_laplace(mu=0, b=1, size=1):
    result = []
    
    for _ in range(size):
        u = random.random()
        if u < 0.5:
            x = mu + b * math.log(2 * u)
        else:
            x = mu - b * math.log(2 * (1 - u))
        
        result.append(x)
    
    return result if size > 1 else result[0]

def estimate_weibull_moments(data):
    n = len(data)
    mean_sample = sum(data) / n
    var_sample = sum((x - mean_sample) ** 2 for x in data) / n

    # Function to solve for α
    def equation(alpha):
        g1 = gamma(1 + 1/alpha)
        g2 = gamma(1 + 2/alpha)
        return (var_sample / (mean_sample ** 2)) - (g2 / (g1 ** 2) - 1)

    # Initial guess for α
    alpha_initial_guess = 1.0
    alpha_solution = fsolve(equation, alpha_initial_guess)[0]

    # Estimate β
    beta_est = mean_sample / gamma(1 + 1 / alpha_solution)

    return alpha_solution, beta_est




if __name__ == "__main__":
    log_norm = generate_laplace(0, 2, 1000)
    print(log_norm)





    