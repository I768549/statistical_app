import random
import math
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



if __name__ == "__main__":
    log_norm = generate_laplace(0, 2, 1000)
    print(log_norm)





    