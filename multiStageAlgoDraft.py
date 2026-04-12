import numpy as np
import random
import math
import copy

class AdaptiveMultiStageHPO:
    def __init__(self, total_budget=30, alpha=0.5, beta=0.5):
        """
        Initializes the optimizer based on the thesis constraints.
        """
        self.B = total_budget
        self.alpha = alpha
        self.beta = beta
        
        # Thesis Allocation Ratios (Table 3)
        self.rho_S = 0.60
        self.rho_R = 0.40
        self.gamma = 0.30
        
        # Calculate Budgets (Equations 9 - 18)
        self.S = int(self.rho_S * self.B)
        self.R = int(self.rho_R * self.B)
        
        # Eq 16: P = floor(gamma * rho_S * B)
        self.P = max(2, int(self.gamma * self.S)) # Ensure at least 2 for crossover
        
        # Eq 17: G = floor(rho_S * B / P)   c
        self.G = max(1, int(self.S / self.P))
        
        print(f"--- HPO Budgets Initialized ---")
        print(f"Total Budget (B): {self.B} trials")
        print(f"Selection Budget (S): {self.S} | Refinement Budget (R): {self.R}")
        print(f"Population Size (P): {self.P} | Generations (G): {self.G}")
        print(f"-------------------------------")

        # Search Space (Table 2)
        self.search_space = {
            'learning_rate': {'type': 'continuous', 'range': [1e-5, 1e-2]},
            'dropout_rate': {'type': 'continuous', 'range': [0.1, 0.5]},
            'activation': {'type': 'categorical', 'values': ['ReLU', 'Tanh', 'ELU']},
            'conv_layers': {'type': 'discrete', 'range': [1, 4]},
            'filter_layers': {'type': 'categorical', 'values': [16, 32, 64, 128]},
            'kernel_size': {'type': 'categorical', 'values': [2, 3, 5, 7]}
        }

    def generate_individual(self):
        """Randomly initializes a hyperparameter configuration vector (Eq 7)."""
        ind = {}
        for key, params in self.search_space.items():
            if params['type'] == 'continuous':
                # Log-uniform for learning rate, uniform for dropout
                if key == 'learning_rate':
                    ind[key] = 10 ** np.random.uniform(np.log10(params['range'][0]), np.log10(params['range'][1]))
                else:
                    ind[key] = np.random.uniform(params['range'][0], params['range'][1])
            elif params['type'] == 'discrete':
                ind[key] = np.random.randint(params['range'][0], params['range'][1] + 1)
            elif params['type'] == 'categorical':
                ind[key] = random.choice(params['values'])
        return ind

    def fitness_function(self, loss, latency):
        """Calculates fitness prioritizing accuracy and latency equally (Eq 19)."""
        # Note: Latency should be normalized or scaled if it outweighs loss in magnitude.
        # Assuming latency is passed in seconds for the calculation, or normalized ms.
        latency_penalty = (latency / 200.0) if latency > 0 else 0 # Normalized against the 200ms strict constraint
        return self.alpha * (1.0 - loss) - self.beta * latency_penalty

    def crossover(self, parent1, parent2):
        """Uniform crossover with 0.5 probability (Eq 30)."""
        offspring = {}
        for key in self.search_space.keys():
            offspring[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return offspring

    def mutate(self, individual):
        """Applies task-dependent mutation with Pm = 0.5 (Eq 31 - 33)."""
        mutated = copy.deepcopy(individual)
        for key, params in self.search_space.items():
            if random.random() < 0.5: # 50% mutation chance per gene
                if params['type'] == 'continuous':
                    # Gaussian mutation (Eq 33)
                    sigma = (params['range'][1] - params['range'][0]) * 0.1
                    new_val = mutated[key] + np.random.normal(0, sigma)
                    mutated[key] = np.clip(new_val, params['range'][0], params['range'][1])
                elif params['type'] == 'discrete':
                    # Random resetting
                    mutated[key] = np.random.randint(params['range'][0], params['range'][1] + 1)
                elif params['type'] == 'categorical':
                    # Random resetting
                    mutated[key] = random.choice(params['values'])
        return mutated

    def run_optimization(self, train_eval_fn):
        """
        Orchestrates the Selection and Refinement stages.
        `train_eval_fn` is a callback that takes (hyperparameters, stress_test_flag) 
        and returns (loss, latency).
        """
        # --- SELECTION STAGE ---
        population = [{'config': self.generate_individual(), 'fitness': 0, 'loss': 0, 'latency': 0} for _ in range(self.P)]
        
        all_losses = [] # Track all losses for P25 calculation later

        for gen in range(self.G):
            print(f"\n--- Generation {gen + 1}/{self.G} ---")
            
            # Evaluate Population
            for ind in population:
                loss, latency = train_eval_fn(ind['config'], stress_test=False)
                ind['loss'] = loss
                ind['latency'] = latency
                ind['fitness'] = self.fitness_function(loss, latency)
                all_losses.append(loss)
            
            # Rank and Select (Eq 28 - 29)
            population.sort(key=lambda x: x['fitness'], reverse=True)
            survivors = population[:max(1, len(population)//2)]
            
            # Breed Next Generation
            next_generation = copy.deepcopy(survivors)
            while len(next_generation) < self.P:
                # Select random parents from survivors
                p1, p2 = random.sample(survivors, 2) if len(survivors) > 1 else (survivors[0], survivors[0])
                offspring_config = self.crossover(p1['config'], p2['config'])
                mutated_config = self.mutate(offspring_config)
                next_generation.append({'config': mutated_config, 'fitness': 0, 'loss': 0, 'latency': 0})
            
            population = next_generation

        # --- REFINEMENT STAGE ---
        print("\n--- Entering Refinement Stage ---")
        # Re-evaluate final population to get exact ranking
        for ind in population:
            if ind['fitness'] == 0: # Only evaluate new offspring
                loss, latency = train_eval_fn(ind['config'], stress_test=False)
                ind['loss'], ind['latency'] = loss, latency
                ind['fitness'] = self.fitness_function(loss, latency)
                all_losses.append(loss)
                
        population.sort(key=lambda x: x['fitness'], reverse=True)
        p25_loss = np.percentile(all_losses, 25) # 25th percentile of all observed losses

        for rank, candidate in enumerate(population):
            if rank >= self.R:
                break # Reached refinement budget limit
                
            print(f"Testing Candidate {rank + 1}...")
            # Evaluate with stress testing (Noise + Latency Jitter activated in callback)
            stress_loss, stress_latency = train_eval_fn(candidate['config'], stress_test=True)
            
            print(f"Candidate {rank + 1} Results -> Loss: {stress_loss:.4f} (Req <= {p25_loss:.4f}), Latency: {stress_latency:.2f}ms (Req <= 200ms)")
            
            # Constraint-aware check (Eq 40)
            if stress_latency <= 200.0 and stress_loss <= p25_loss:
                print(f"\n>>> OPTIMAL CONFIGURATION FOUND (Candidate {rank + 1}) <<<")
                return candidate['config']
            else:
                print("Candidate failed constraints. Backtracking to next best...")

        print("\n>>> NO CANDIDATE PASSED ALL STRICT CONSTRAINTS. RETURNING BEST BEST-EFFORT CONFIG <<<")
        return population[0]['config']
