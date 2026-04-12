import numpy as np
import random
import math
import copy

class AdaptiveMultiStageStressTestedHPO:
    def __init__ (self, 
                 budgetTrial = 30, 
                 alpha = 0.5,   #accuracy priority
                 beta = 0.5,    #latency priority
                 rho_S = 0.6,   #selection stage budget trial
                 rho_R = 0.60,  #refinement stage budget trial
                 gamma = 0.3    #allocation ratio
                 ):
        
        self.B = budgetTrial        # Budget Trial
        self.alpha = alpha          # Accuracy priority factor
        self.beta = beta            # Latency priority factor

        self.S = int(rho_S * self.B)    # Selection stage budget
        self.R = int(rho_R * self.B)    # Refinement stage budget
        self.gamma_alloc = gamma        # Population allocation ratio

        self.P = max(2, math.ceil(self.gamma * self.S)) # Populations
        self.G = max(1, math.floor(self.S / self.P))    # Generations

        print(f"--- HPO Budgets Initialized ---")
        print(f"Total Budget (B): {self.B} trials")
        print(f"Selection Budget (S): {self.S} | Refinement Budget (R): {self.R}")
        print(f"Population Size (P): {self.P} | Generations (G): {self.G}")
        print(f"-------------------------------")

        self.search_space = {
            'learning_rate': {'type': 'continuous',                         # Logarithmic scale to capture fine to coarse weight updates.
                              'range': [1e-5, 1e-2]},
            'dropout_rate': {'type': 'comtinuous',                          # To prevent overfitting without excessive information loss.
                             'range' : [0.1, 0.5]},
            'acivation_function' : {'type': 'categorical',                  # To test non-linearity performance in spectral feature maps.
                                    'values': ['ReLU', 'Tanh', 'ELU']},
            'convolution_layers': {'type': 'discrete',                      # Balancing depth for accuracy vs. latency
                                   'range': [1, 4]},
            'filter_layers': {'type': 'categorical',                        # Standard power-of-two filter counts for efficient memory alignment
                              'values': [16, 32, 64, 128]},
            'filter_size': {'type': 'categorical',                          # Typical kernel sizes for capturing local and regional spectral patterns.
                            'values': [2, 3, 5, 7]}
        }

    def individual_generation (self):
        individual = {}

        for key, parameter in self.search_space.items():
            if parameter['type'] == 'continuous':
                if key == 'learning_rate':
                    individual[key] = 10 ** np.random.uniform(np.log10(parameter['range']['0']), 
                                                              np.log10(parameter['range']['1']))
                else:
                    individual[key] = np.random.uniform(parameter['range'][0],
                                                        parameter['range'][1])
            elif parameter['type'] == 'discrete':
                individual[key] = np.random.uniform(parameter['range'][0], 
                                                    parameter['range'][1] + 1)
            elif parameter['type'] == 'categorical':
                individual[key] = random.choice(parameter['values'])
    
    def fitness_function (self, loss, latency):
        
        latency_penalty = (latency/200) if latency > 0 else 0
        return self.alpha * (1 - loss) - self.beta * (latency_penalty)

    def crossover (self,
                   parent1,
                   parent2):
        offspring = {}

        for key in self.search_space.keys():
            offspring[key] = parent1[key] if random < 0.5 else parent2[key]
        return offspring
    
    def mutate (self, 
                individual):
        
        mutated = copy.deepcopy(individual)
        
        for key, parameter in self.search_space.items():
            if random.random() < 0.5:

                if parameter['type'] == 'continuous':            
                    sigma = (parameter['range'][1] - parameter['range'][0]) * 0.1
                    new_val = mutated[key] + np.random.normal(0, sigma)
                    mutated[key] = np.clip(new_val,
                                           parameter['range'][0],
                                           parameter['range'][1])
                    
                elif parameter['type'] == 'discrete':
                    mutated[key] = np.random.randint(parameter['range'][0], parameter['range']['1'] + 1)

                elif parameter['type'] == 'categorical':
                    mutated[key] = random.choice(parameter['values'])

        return mutated
    
    def optimization (self, train_eval):
        # SELECTION STAGE #
        population = [{'config': self.individual_generatio, 
                         'fitness': 0, 
                         'latency': 0, 
                         'loss': 0} for population in range(self.P)]
        total_loss = []

        for generations in range(self.G):
            for individual in population:
                loss, latency, = train_eval(individual['config'], stress_test = False)
                individual['loss'] = loss
                individual['latency'] = latency
                individual['fitness'] = self.fitness_function(loss, latency)
                total_loss.append(loss)\
                
            population.sort(key=lambda x:x['fitness'], reverse=True)
            survivors = population[:max(1, len(population)//2)]

            next_gen = copy.deepcopy(survivors)
            while len(next_gen) < self.P:
                p1, p2 = random.sample(survivors, 2) if len(survivors) > 1 else (survivors[0], survivors[0])
                offspring = self.crossover(p1['config'], p2['config'])
                mutated = self.mutate(offspring)
                next_gen.append({'config': mutated, 'fitness': 0, 'latency': 0, 'loss': 0})

            population = next_gen

        # REFINEMENT STAGE #

        for ind in population:
            if ind['fitness'] == 0:
                loss, latency = train_eval(ind['config'], stressTest = False)
                ind['loss'] , ind['latency'] = loss, latency
                ind['fitness'] = self.fitness_function(loss, latency)
                total_loss.append(loss)