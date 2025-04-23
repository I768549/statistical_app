import random
import math
def generate_exp_theoretical_dist(size, lam = 1):
    dist_array = []
    for i in range(size):
        #zero_one_array.append(random.random())
        dist_array.append((math.log(1 - random.random())/-lam))
    return dist_array

if __name__ == "__main__":
    print(generate_exp_theoretical_dist(5, 2))