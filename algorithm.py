import numpy as np
import random
import math
import copy
import os
import csv
import pickle

class AdaptiveMultiStageStressTestedHPO:
    def __init__ (self, 
                 budgetTrial, 
                 alpha,   #accuracy priority
                 beta,    #latency priority
                 rho_S,   #selection stage budget trial
                 rho_R,  #refinement stage budget trial
                 gamma ,    #allocation ratio
                 logfile,
                 checkpoint
                 ):
        
        self.B = budgetTrial        # Budget Trial
        self.alpha = alpha          # Accuracy priority factor
        self.beta = beta            # Latency priority factor

        self.S = int(rho_S * self.B)    # Selection stage budget
        self.R = int(rho_R * self.B)    # Refinement stage budget
        self.gamma_alloc = gamma        # Population allocation ratio

        self.P = max(2, math.ceil(self.gamma_alloc * self.S)) # Populations
        self.G = max(1, math.floor(self.S / self.P))    # Generations

        print(f"--- HPO Budgets Initialized ---")
        print(f"Total Budget (B): {self.B} trials")
        print(f"Selection Budget (S): {self.S} | Refinement Budget (R): {self.R}")
        print(f"Population Size (P): {self.P} | Generations (G): {self.G}")
        print(f"-------------------------------")

        self.search_space = {
            'learning_rate': {'type': 'continuous',                         # Logarithmic scale to capture fine to coarse weight updates.
                              'range': [1e-5, 1e-2]},
            'dropout_rate': {'type': 'continuous',                          # To prevent overfitting without excessive information loss.
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
        self.logfile = logfile
        self._init_csv_logger()

        self.checkpoint_file = checkpoint

    def _init_csv_logger(self):
        file_exists = os.path.isfile(self.logfile)
        with open(self.logfile, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage', 
                             'Gen or Rank', 
                             'Learning Rate', 
                             'Activation Function',
                             '# of Convolution Layers',
                             '# of Filter Layers',
                             "Kernel Size",
                             'Loss Value',
                             'Latency',
                             'Fitness Score',
                             'Status'])

    def individual_logging(self, 
                           stage, 
                           gen_rank, 
                           config, 
                           loss, 
                           latency, 
                           fitness, 
                           status):
        with open(self.logfile, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([stage,
                             gen_rank,
                             f"{config['learning_rate']:.6f}",
                             config['dropout_rate'],
                             config['activation_function'],
                             config['convolution_layers'],
                             config['filter_layers'],
                             config['filter_size'],
                             f"{loss:.4f}",
                             f"{latency:.2f}",
                             f"{fitness:.4f}",
                             status        
                            ])

    def save_checkpoint(self, state):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)

        print(f"Checkpoint file saved at {self.checkpoint_file}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            print(f"Checkpoint file loaded. Continue saved progress")
            return state
        return None

    def clear_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print("Checkpoint cleared. Optimization done.")

    def individual_generation (self):
        individual = {}

        for key, parameter in self.search_space.items():
            if parameter['type'] == 'continuous':
                if key == 'learning_rate':
                    individual[key] = 10 ** np.random.uniform(np.log10(parameter['range'][0]), 
                                                              np.log10(parameter['range'][1]))
                else:
                    individual[key] = np.random.uniform(parameter['range'][0],
                                                        parameter['range'][1])
            elif parameter['type'] == 'discrete':
                individual[key] = np.random.randint(parameter['range'][0], 
                                                    parameter['range'][1] + 1)
            elif parameter['type'] == 'categorical':
                individual[key] = random.choice(parameter['values'])
    
    def fitness_function (self, 
                          loss, 
                          latency):
        
        latency_penalty = (latency/200) if latency > 0 else 0
        return self.alpha * (1 - loss) - self.beta * (latency_penalty)

    def crossover (self,
                   parent1,
                   parent2):
        offspring = {}

        for key in self.search_space.keys():
            offspring[key] = parent1[key] if random.random() < 0.5 else parent2[key]
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
                    mutated[key] = np.random.randint(parameter['range'][0], parameter['range'][1] + 1)

                elif parameter['type'] == 'categorical':
                    mutated[key] = random.choice(parameter['values'])

        return mutated
    
    def optimization (self, train_eval):
        state = self.load_checkpoint()

        if state:
            stage = state['state']
            population = state['population']
            total_loss = state['total_loss']
            start_gen = state.get('generation', 0)
            refine_rank = state.get('refine_rank', 0)
            print(f"Resuming optimization from Stage: {stage}, Population: {population}, Generation: {start_gen}, Rank (Refinement Stage): {refine_rank}")
        else:
            print(f"Entering selection stage")
            stage = 'Selection'
            population = [{'config': self.individual_generation, 
                            'fitness': 0, 
                            'latency': 0, 
                            'loss': 0} for population in range(self.P)]
            total_loss = []
            start_gen = 0
            refine_rank = 0

        if stage == 'Selection':
            for generations in range(self.G):
                for individual in population:
                    loss, latency, = train_eval(individual['config'], stress_test = False)
                    individual['loss'] = loss
                    individual['latency'] = latency
                    individual['fitness'] = self.fitness_function(loss, latency)
                    total_loss.append(loss)
                    
                    self.individual_logging("Selection", 
                                            f"Generation {generations + 1}", 
                                            ind['config'], 
                                            loss, 
                                            latency, 
                                            ind['fitness'],
                                            "Evaluated")
                    
                population.sort(key=lambda x:x['fitness'], reverse=True)
                survivors = population[:max(1, len(population)//2)]

                #   Crossover and Mutation
                next_gen = copy.deepcopy(survivors)
                while len(next_gen) < self.P:
                    p1, p2 = random.sample(survivors, 2) if len(survivors) > 1 else (survivors[0], survivors[0])
                    offspring = self.crossover(p1['config'], p2['config'])
                    mutated = self.mutate(offspring)
                    next_gen.append({'config': mutated, 'fitness': 0, 'latency': 0, 'loss': 0})

                population = next_gen

                self.save_checkpoint({
                    'stage': 'selection',
                    'generation': generations + 1,
                    'population': population,
                    'total_loss': total_loss
                })

            #   Evaluation of Final Generation for Refinement Stage
            stage = 'Refinement'

            for ind in population:
                if ind['fitness'] == 0:
                    loss, latency = train_eval(ind['config'], stress_Test = False)
                    ind['loss'] , ind['latency'] = loss, latency
                    ind['fitness'] = self.fitness_function(loss, latency)
                    total_loss.append(loss)

            population.sort(key=lambda x: x['fitness'], reverse=True)
            self.save_checkpoint({
                'stage': 'Refinement',
                'population': population,
                'total_loss': total_loss,
                'refine_rank': 0
            })
        
        if stage == 'Refinement':
            p25_loss = np.percentile(total_loss, 25) 

            for rank in range(min(self.R, len(population))):
                candidate = population[rank]
                print(f"Testing candidate {rank + 1} (Fitness score: {candidate['fitness']:.4f})")

                #  ROBUST-BASED STRESS TESTING (SIMULATING HARDWARE NOISE)
                stress_loss, stress_latency = train_eval(candidate['config'], stressTest = True)

                #   LATENCY JITTER
                p_trigger = 0.10
                if np.random.rand() < p_trigger:
                    jitter = np.random.uniform(0, 50)
                    stress_latency += jitter
                    print(f"Added {jitter:.2f}ms latency jitter")

                self.save_checkpoint({
                    'stage': 'Refinement',
                    'population': population,
                    'total_loss': total_loss,
                    'refine_rank': rank + 1
                })

                #   CONSTRAINT-AWARE REFINEMENT
                if stress_latency <= 200.0 and stress_loss <= p25_loss:
                    print(f"ACCEPT THE INDIVIDUAL as the optimal hyperparameter")
                    self.individual_logging("Refinement", 
                                            f"Rank {rank + 1}", 
                                            candidate['config'], 
                                            stress_loss, 
                                            stress_latency, 
                                            candidate['fitness'],
                                            "Promoted as Optimal")
                    return candidate['config']
                else:
                    print(f"Individual failed the test. Kill the individual. Promote second best individual to refinement stage")

        print("Refinement budget trial used up. No candidate passed")
        self.individual_logging("Refinement", 
                                "Fallback", 
                                population[0]['config'], 
                                population[0]['loss'],
                                population[0]['latency'],
                                population[0]['fitness'],
                                "Fallback")
