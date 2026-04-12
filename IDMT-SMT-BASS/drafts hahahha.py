import torch
import torch.optim as optim
import time
import numpy as np
import optuna

# Import your custom mathematical engine and model
from adaptive_hpo import AdaptiveMultiStageHPO
from tabcnn_model import DynamicTabCNN

# ==========================================
# 1. THE SHARED PYTORCH TRAINING PIPELINE
# ==========================================
def evaluate_model_pipeline(config, stress_test=False):
    """
    Ito ang NAG-IISANG training loop para sa thesis mo.
    Gagamitin ito pareho ng Proposed Algorithm mo at ng Optuna (Random/Bayesian).
    Garantisadong fair ang comparison!
    """
    model = DynamicTabCNN(config)
    # Ilipat agad sa GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
        
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # --- SIMULATED TRAINING LOOP ---
    # (Ilalagay mo rito ang totoong PyTorch dataloader for loop mo)
    # ...
    
    # --- EVALUATION AND LATENCY CALCULATION (Eq 20) ---
    model.eval()
    dummy_input = torch.randn(1, 4608) # Raw audio 209ms context window (18 frames * 256 hop)
    if torch.cuda.is_available():
        dummy_input = dummy_input.to('cuda')
        
    latencies = []
    
    with torch.no_grad():
        for _ in range(10): # Simulate evaluating 10 frames
            t_i = time.perf_counter()
            _ = model(dummy_input) # Forward pass (kasama na CQT sa loob)
            t_o = time.perf_counter()
            
            delta_t_ms = (t_o - t_i) * 1000
            latencies.append(delta_t_ms)
            
    avg_latency = sum(latencies) / len(latencies)
    validation_loss = np.random.uniform(0.1, 0.4) # Simulated validation loss
    
    # --- STRESS TESTING (Para lang sa Proposed Algo Refinement Stage) ---
    if stress_test:
        validation_loss += np.random.uniform(0.05, 0.1)
        if np.random.rand() < 0.10: 
            avg_latency += np.random.uniform(0, 50)
            
    return validation_loss, avg_latency

# ==========================================
# 2. OPTUNA OBJECTIVE WRAPPER
# ==========================================
def optuna_objective(trial):
    """
    Ito ang tulay para maintindihan ni Optuna yung config format natin.
    """
    # 1. I-define ang search space ni Optuna (Gagayahin yung Table 2 mo)
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'activation': trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'ELU']),
        'conv_layers': trial.suggest_int('conv_layers', 1, 4),
        'filter_layers': trial.suggest_categorical('filter_layers', [16, 32, 64, 128]),
        'kernel_size': trial.suggest_categorical('kernel_size', [2, 3, 5, 7])
    }
    
    # 2. I-run ang SHARED PyTorch pipeline (Walang stress test ang Optuna)
    loss, latency = evaluate_model_pipeline(config, stress_test=False)
    
    # 3. I-compute ang fitness gamit ang parehong weights mo (Alpha 0.5, Beta 0.5)
    # Dahil MINIMIZE ang ginagawa ng Optuna by default, ibabato natin ang NEGATIVE fitness
    # o kaya yung mismong raw loss at i-constraint na lang ang latency.
    # Para pareho sa custom algo mo, kukunin natin yung inverse ng Eq 19:
    latency_penalty = (latency / 200.0) if latency > 0 else 0 
    fitness = 0.5 * (1.0 - loss) - 0.5 * latency_penalty
    
    # Optuna maximizes this if we set direction="maximize"
    return fitness

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Thesis Optimization Experiments ---")
    
    # Piliin kung anong algorithm ang ite-test mo ngayon:
    # Choices: "PROPOSED", "RANDOM", "BAYESIAN"
    MODE = "PROPOSED" 
    
    TOTAL_TRIALS = 30 # Standardized budget (Table 2 ng thesis mo)
    
    if MODE == "PROPOSED":
        print("\n>>> RUNNING PROPOSED ADAPTIVE HPO <<<")
        hpo_engine = AdaptiveMultiStageHPO(total_budget=TOTAL_TRIALS)
        best_config = hpo_engine.run_optimization(evaluate_model_pipeline)
        print("\nBest Config from Proposed Algo:", best_config)
        save_name = "trained_prop.pth"

    elif MODE == "RANDOM":
        print("\n>>> RUNNING OPTUNA RANDOM SEARCH <<<")
        # I-set ang sampler sa RandomSampler
        sampler = optuna.samplers.RandomSampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(optuna_objective, n_trials=TOTAL_TRIALS)
        best_config = study.best_params
        print("\nBest Config from Random Search:", best_config)
        save_name = "trained_random.pth"

    elif MODE == "BAYESIAN":
        print("\n>>> RUNNING OPTUNA BAYESIAN OPTIMIZATION (TPE) <<<")
        # I-set ang sampler sa TPESampler (Ito yung default Bayesian method ng Optuna)
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(optuna_objective, n_trials=TOTAL_TRIALS)
        best_config = study.best_params
        print("\nBest Config from Bayesian Opt:", best_config)
        save_name = "trained_bayesian.pth"

    # ==========================================
    # 4. FINAL RETRAINING AND SAVING
    # ==========================================
    print(f"\n>>> RETRAINING FINAL MODEL USING BEST {MODE} CONFIG <<<")
    # I-instantiate ang model gamit ang winning hyperparameters
    final_model = DynamicTabCNN(best_config)
    if torch.cuda.is_available():
        final_model = final_model.to('cuda')
        
    # Dito mo patatakbuhin yung final full training loop mo (e.g., 50-100 epochs)
    # gamit yung buong training dataset para makuha yung pinakamataas na accuracy.
    # ... (Ilalagay ang PyTorch training loop dito) ...
    
    # I-save ang trained weights sa specific filename
    torch.save(final_model.state_dict(), save_name)
    
    # I-save din yung config as JSON para madaling basahin ng deployment.py sa Jetson Nano
    import json
    with open(save_name.replace('.pth', '_config.json'), 'w') as f:
        json.dump(best_config, f, indent=4)
        
    print(f"✅ Final model weights saved to: {save_name}")
    print(f"✅ Final model config saved to: {save_name.replace('.pth', '_config.json')}")