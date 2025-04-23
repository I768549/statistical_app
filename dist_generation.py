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

if __name__ == "__main__":
    print(generate_exp_theoretical_dist(5, 2))