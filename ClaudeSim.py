#!/usr/bin/env python
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import itertools
import os
import gc
import time
from copy import deepcopy

# Import your model classes
from DisasterModelNew import DisasterModel, HumanAgent, AIAgent

# Ensure output directory exists
os.makedirs("simulation_outputs", exist_ok=True)
os.makedirs("simulation_outputs/temp_results", exist_ok=True)

# Define experiment configurations
def run_experiments():
    # Parameter ranges to test
    experiment_params = {
        'share_exploitative': [0.2, 0.5, 0.8],           # Varying exploratory/exploitative ratio
        'share_of_disaster': [0.1, 0.2, 0.3],            # Disaster coverage
        'disaster_dynamics': [1, 2, 3],                  # How quickly disaster changes
        'ai_alignment_level': [0.0, 0.5, 1.0],           # How much AI adapts to human beliefs
        'initial_ai_trust': [0.4, 0.7, 0.9]              # Initial trust in AI
    }
    
    # Fixed parameters
    fixed_params = {
        'initial_trust': 0.5,               # Initial trust between humans
        'number_of_humans': 50,             # Total human population
        'share_confirming': 0.5,            # Fraction of humans with confirming attitudes
        'shock_probability': 0.1,           # Probability of disaster shocks
        'shock_magnitude': 2,               # Magnitude of disaster shocks
        'trust_update_mode': "average",
        'exploitative_correction_factor': 1.0,
        'width': 50, 
        'height': 50,
        'ticks': 600                        # Simulation duration
    }
    
    # Generate all combinations to test
    param_combinations = list(itertools.product(
        experiment_params['share_exploitative'],
        experiment_params['share_of_disaster'],
        experiment_params['disaster_dynamics'],
        experiment_params['ai_alignment_level'],
        experiment_params['initial_ai_trust']
    ))
    
    # Process in smaller batches to manage memory
    batch_size = 5  # Adjust this based on your system capabilities
    num_batches = (len(param_combinations) + batch_size - 1) // batch_size
    
    print(f"Running {len(param_combinations)} experiments in {num_batches} batches")
    
    # Create a temporary file to track completion
    with open("simulation_outputs/experiment_progress.txt", "w") as f:
        f.write(f"0/{len(param_combinations)} complete\n")
    
    # Process batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(param_combinations))
        batch = param_combinations[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches} ({len(batch)} parameter combinations)")
        
        # Run batch in parallel
        num_processes = min(mp.cpu_count(), len(batch))
        chunks = np.array_split(batch, num_processes)
        
        with mp.Pool(processes=num_processes) as pool:
            pool.map(process_chunk, [(chunk_idx, chunk, fixed_params, experiment_params) 
                                    for chunk_idx, chunk in enumerate(chunks)])
        
        # Force garbage collection
        gc.collect()
        
        # Update progress
        total_processed = end_idx
        with open("simulation_outputs/experiment_progress.txt", "w") as f:
            f.write(f"{total_processed}/{len(param_combinations)} complete\n")
    
    # Combine all results files
    combine_result_files()
    
    # Generate analysis and visualizations
    results_df = pd.read_csv("simulation_outputs/experiment_results.csv")
    analyze_results(results_df, experiment_params)
    
    # Clean up temporary files
    for file in os.listdir("simulation_outputs/temp_results"):
        os.remove(os.path.join("simulation_outputs/temp_results", file))

def process_chunk(args):
    chunk_idx, chunk, fixed_params, param_names = args
    
    # Create a temp filename for this chunk's results
    temp_filename = f"simulation_outputs/temp_results/chunk_{chunk_idx}_{int(time.time())}.csv"
    
    # Process each parameter combination
    for param_idx, params in enumerate(chunk):
        # Create parameter dictionary for this run
        run_params = {
            'share_exploitative': params[0],
            'share_of_disaster': params[1],
            'disaster_dynamics': params[2],
            'ai_alignment_level': params[3],
            'initial_ai_trust': params[4],
            **fixed_params  # Add fixed parameters
        }
        
        # Run single experiment
        result = run_single_experiment(run_params)
        
        # Convert to DataFrame
        result_df = pd.DataFrame([result])
        
        # Write to temp file (append mode if not the first result)
        mode = 'a' if param_idx > 0 else 'w'
        header = not (param_idx > 0)
        result_df.to_csv(temp_filename, index=False, mode=mode, header=header)
        
        # Free memory
        del result
        del result_df
        gc.collect()

def combine_result_files():
    """Combine all temporary result files into a single CSV"""
    all_dfs = []
    temp_dir = "simulation_outputs/temp_results"
    
    for filename in os.listdir(temp_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(temp_dir, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)
    
    # Combine and save
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv("simulation_outputs/experiment_results.csv", index=False)
        
        # Free memory
        del all_dfs
        del combined_df
        gc.collect()

def run_single_experiment(params):
    # Extract parameters for this run
    ticks = params.pop('ticks')  # Remove ticks from params for model init
    ai_alignment_level = params.pop('ai_alignment_level')  # Remove for model init
    
    # Create model instance with the specified parameters
    model = DisasterModel(**params)
    
    # Set AI alignment level
    model.ai_alignment_level = ai_alignment_level
    
    # Run simulation
    for _ in range(ticks):
        model.step()
    
    # Calculate metrics
    metrics = calculate_metrics(model)
    
    # Add parameters to metrics for result analysis
    result = {
        'share_exploitative': params['share_exploitative'],
        'share_of_disaster': params['share_of_disaster'],
        'disaster_dynamics': params['disaster_dynamics'],
        'ai_alignment_level': ai_alignment_level,
        'initial_ai_trust': params['initial_ai_trust'],
        **metrics
    }
    
    # Free up memory
    del model
    
    return result

def calculate_metrics(model):
    # Calculate echo chamber metrics
    human_belief_variance = calculate_belief_variance(model, "human_only")
    human_ai_belief_variance = calculate_belief_variance(model, "human_ai")
    
    # Calculate assistance effectiveness
    assistance_correct = sum([model.assistance_exploit.get(pos, 0) + model.assistance_explor.get(pos, 0) 
                              for pos, level in model.disaster_grid.items() if level >= 4])
    
    # Total cells in need
    cells_in_need = sum([1 for _, level in model.disaster_grid.items() if level >= 4])
    assistance_coverage = assistance_correct / max(1, cells_in_need)
    
    # Calculate incorrect assistance
    assistance_incorrect = sum([model.assistance_incorrect_exploit.get(pos, 0) + model.assistance_incorrect_explor.get(pos, 0) 
                               for pos, level in model.disaster_grid.items() if level <= 2])
    
    # Total cells not in need
    cells_not_in_need = sum([1 for _, level in model.disaster_grid.items() if level <= 2])
    incorrect_rate = assistance_incorrect / max(1, cells_not_in_need)
    
    # Trust metrics - last 100 ticks
    recent_trust = model.trust_data[-100:]
    avg_exp_human_trust = np.mean([t[0] for t in recent_trust])
    avg_exp_ai_trust = np.mean([t[1] for t in recent_trust])
    avg_expl_human_trust = np.mean([t[2] for t in recent_trust])
    avg_expl_ai_trust = np.mean([t[3] for t in recent_trust])
    
    # Calculate call rates - last 100 ticks
    recent_calls = model.calls_data[-100:]
    call_ratio_exp_human_ai = sum([c[0] for c in recent_calls]) / max(1, sum([c[1] for c in recent_calls]))
    call_ratio_expl_human_ai = sum([c[2] for c in recent_calls]) / max(1, sum([c[3] for c in recent_calls]))
    
    # Calculate unmet needs statistics
    avg_unmet_needs = np.mean(model.unmet_needs_evolution[-100:])
    
    return {
        'human_echo_chamber': human_belief_variance,
        'human_ai_echo_chamber': human_ai_belief_variance,
        'assistance_coverage': assistance_coverage,
        'incorrect_assistance_rate': incorrect_rate,
        'avg_exp_human_trust': avg_exp_human_trust,
        'avg_exp_ai_trust': avg_exp_ai_trust,
        'avg_expl_human_trust': avg_expl_human_trust,
        'avg_expl_ai_trust': avg_expl_ai_trust,
        'call_ratio_exp_human_ai': call_ratio_exp_human_ai,
        'call_ratio_expl_human_ai': call_ratio_expl_human_ai,
        'avg_unmet_needs': avg_unmet_needs
    }

def calculate_belief_variance(model, mode="human_only"):
    """
    Calculate variance in beliefs as a proxy for echo chambers
    Higher variance = less echo chamber effect
    """
    belief_differences = []
    
    # For each cell, calculate differences in agent beliefs
    for cell in model.disaster_grid.keys():
        cell_beliefs = []
        for agent in model.humans.values():
            cell_beliefs.append(agent.beliefs.get(cell, 0))
        
        if mode == "human_ai" and model.ais:
            # Add AI sensed data
            for ai_agent in model.ais.values():
                if cell in ai_agent.sensed:
                    cell_beliefs.append(ai_agent.sensed[cell])
        
        # Skip if not enough data points
        if len(cell_beliefs) <= 1:
            continue
            
        # Calculate variance for this cell
        cell_variance = np.var(cell_beliefs)
        belief_differences.append(cell_variance)
    
    # Lower value = more echo chamber (less variance in beliefs)
    if belief_differences:
        return np.mean(belief_differences)
    return 0

def analyze_results(results_df, experiment_params):
    """Analyze and visualize the results of all experiments"""
    
    # Create directory for plots
    plot_dir = "simulation_outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Process in batches to save memory
    metrics_to_plot = [
        'human_echo_chamber', 'assistance_coverage', 
        'incorrect_assistance_rate', 'avg_unmet_needs',
        'human_ai_echo_chamber', 'avg_exp_ai_trust', 
        'avg_expl_ai_trust'
    ]
    
    # 1. Impact of share_exploitative on key metrics
    for metric in metrics_to_plot:
        create_boxplot(results_df, 'share_exploitative', metric, 
                      f"Impact of Exploitative/Exploratory Ratio on {metric}", 
                      plot_dir)
        # Free memory
        gc.collect()
    
    # 2. Impact of ai_alignment_level on key metrics
    for metric in metrics_to_plot:
        create_boxplot(results_df, 'ai_alignment_level', metric, 
                      f"Impact of AI Alignment Level on {metric}", 
                      plot_dir)
        # Free memory
        gc.collect()
    
    # 3. Impact of disaster_dynamics on key metrics
    for metric in ['assistance_coverage', 'incorrect_assistance_rate', 'avg_unmet_needs']:
        create_boxplot(results_df, 'disaster_dynamics', metric, 
                      f"Impact of Disaster Dynamics on {metric}", 
                      plot_dir)
        # Free memory
        gc.collect()
    
    # 4. Impact of share_of_disaster on key metrics
    for metric in ['assistance_coverage', 'incorrect_assistance_rate', 'avg_unmet_needs']:
        create_boxplot(results_df, 'share_of_disaster', metric, 
                      f"Impact of Disaster Coverage on {metric}", 
                      plot_dir)
        # Free memory
        gc.collect()
    
    # 5. Compare human vs AI trust across agent types (smaller chunks)
    plot_trust_comparison(results_df, plot_dir)
    gc.collect()
    
    # 6. Heatmap of key interactions
    for metric in ['assistance_coverage', 'human_ai_echo_chamber']:
        plot_interaction_heatmap(results_df, 'ai_alignment_level', 'share_exploitative', 
                              metric, plot_dir, f"{metric}_heatmap")
        gc.collect()
    
    # 7. Plot call ratios (human vs AI) for different agent types
    plot_call_ratios(results_df, plot_dir)
    gc.collect()

def create_boxplot(df, x_param, y_param, title, plot_dir):
    """Create a single boxplot and free memory"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_param, y=y_param, data=df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{x_param}_{y_param}_boxplot.png")
    plt.close()

def plot_interaction_heatmap(df, x_param, y_param, metric, plot_dir, filename):
    """Create heatmap showing interaction effects between two parameters"""
    # Group and aggregate data
    pivot_data = df.groupby([x_param, y_param])[metric].mean().reset_index()
    pivot_table = pivot_data.pivot(index=y_param, columns=x_param, values=metric)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".3f")
    plt.title(f"Interaction Effect of {x_param} and {y_param} on {metric}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{filename}.png")
    plt.close()
    
    # Free memory
    del pivot_data
    del pivot_table

def plot_trust_comparison(df, plot_dir):
    """Plot comparison of trust levels across agent types and trust targets"""
    # Prepare data for plotting
    trust_data = pd.DataFrame({
        'Agent Type': ['Exploitative']*len(df) + ['Exploratory']*len(df),
        'Trust Target': ['Human']*len(df) + ['AI']*len(df) + ['Human']*len(df) + ['AI']*len(df),
        'Trust Level': df['avg_exp_human_trust'].tolist() + df['avg_exp_ai_trust'].tolist() + 
                      df['avg_expl_human_trust'].tolist() + df['avg_expl_ai_trust'].tolist(),
        'AI Alignment': df['ai_alignment_level'].tolist()*4
    })
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Agent Type', y='Trust Level', hue='Trust Target', data=trust_data)
    plt.title("Trust Levels by Agent Type and Trust Target")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/trust_comparison.png")
    plt.close()
    
    # Also plot by AI alignment level
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='AI Alignment', y='Trust Level', hue='Trust Target', data=trust_data)
    plt.title("Trust Levels by AI Alignment Level and Trust Target")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/trust_by_ai_alignment.png")
    plt.close()
    
    # Free memory
    del trust_data

def plot_call_ratios(df, plot_dir):
    """Plot call ratios (human:AI) for different agent types"""
    # Prepare data
    call_data = pd.DataFrame({
        'Agent Type': ['Exploitative']*len(df) + ['Exploratory']*len(df),
        'Human:AI Call Ratio': df['call_ratio_exp_human_ai'].tolist() + df['call_ratio_expl_human_ai'].tolist(),
        'Share Exploitative': df['share_exploitative'].tolist()*2,
        'AI Alignment': df['ai_alignment_level'].tolist()*2
    })
    
    # Plot by agent type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Agent Type', y='Human:AI Call Ratio', data=call_data)
    plt.title("Ratio of Human to AI Information Calls by Agent Type")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/call_ratio_by_agent_type.png")
    plt.close()
    
    # Plot by share_exploitative
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Share Exploitative', y='Human:AI Call Ratio', hue='Agent Type', data=call_data)
    plt.title("Human:AI Call Ratio by Population Composition")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/call_ratio_by_population.png")
    plt.close()
    
    # Plot by AI alignment
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='AI Alignment', y='Human:AI Call Ratio', hue='Agent Type', data=call_data)
    plt.title("Human:AI Call Ratio by AI Alignment Level")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/call_ratio_by_ai_alignment.png")
    plt.close()
    
    # Free memory
    del call_data

# Function to run a time-series experiment for a specific configuration
def run_time_series_experiment(config_name, params):
    """Run a single experiment and save time series data"""
    # Extract parameters for this run
    ticks = params.pop('ticks')  # Remove ticks from params for model init
    ai_alignment_level = params.pop('ai_alignment_level')  # Remove for model init
    
    # Create model instance
    model = DisasterModel(**params)
    model.ai_alignment_level = ai_alignment_level
    
    # Create output directory
    os.makedirs("simulation_outputs/time_series", exist_ok=True)
    output_file = f"simulation_outputs/time_series/{config_name}.csv"
    
    # Open file for writing
    with open(output_file, 'w') as f:
        # Write header
        header = "tick,exp_human_trust,exp_ai_trust,expl_human_trust,expl_ai_trust,"
        header += "calls_exp_human,calls_exp_ai,calls_expl_human,calls_expl_ai,"
        header += "unmet_needs,human_echo_chamber,human_ai_echo_chamber\n"
        f.write(header)
        
        # Run simulation and collect time series data
        for i in range(ticks):
            model.step()
            
            # Calculate metrics at this tick
            human_echo = calculate_belief_variance(model, "human_only")
            human_ai_echo = calculate_belief_variance(model, "human_ai")
            
            # Format row data
            row = f"{i},{model.trust_data[-1][0]},{model.trust_data[-1][1]},"
            row += f"{model.trust_data[-1][2]},{model.trust_data[-1][3]},"
            row += f"{model.calls_data[-1][0]},{model.calls_data[-1][1]},"
            row += f"{model.calls_data[-1][2]},{model.calls_data[-1][3]},"
            row += f"{model.unmet_needs_evolution[-1]},{human_echo},{human_ai_echo}\n"
            
            # Write row
            f.write(row)
    
    # Read file back for plotting
    df = pd.read_csv(output_file)
    
    # Create key plots
    plot_time_series(df, config_name)
    
    # Free memory
    del model
    gc.collect()
    
    return df

def plot_time_series(df, config_name):
    """Create time series plots for a specific configuration"""
    plot_dir = "simulation_outputs/time_series/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Trust evolution
    plt.figure(figsize=(12, 8))
    plt.plot(df['tick'], df['exp_human_trust'], label="Exploitative: Human Trust")
    plt.plot(df['tick'], df['exp_ai_trust'], label="Exploitative: AI Trust")
    plt.plot(df['tick'], df['expl_human_trust'], label="Exploratory: Human Trust")
    plt.plot(df['tick'], df['expl_ai_trust'], label="Exploratory: AI Trust")
    plt.xlabel("Tick")
    plt.ylabel("Average Trust")
    plt.title(f"Trust Evolution by Agent Type - {config_name}")
    plt.legend()
    plt.savefig(f"{plot_dir}/{config_name}_trust_evolution.png")
    plt.close()
    gc.collect()
    
    # 2. Information calls
    plt.figure(figsize=(12, 8))
    plt.plot(df['tick'], df['calls_exp_human'], label="Exploitative: Calls to Humans")
    plt.plot(df['tick'], df['calls_exp_ai'], label="Exploitative: Calls to AI")
    plt.plot(df['tick'], df['calls_expl_human'], label="Exploratory: Calls to Humans")
    plt.plot(df['tick'], df['calls_expl_ai'], label="Exploratory: Calls to AI")
    plt.xlabel("Tick")
    plt.ylabel("Information Requests")
    plt.title(f"Information Request Calls by Agent Type - {config_name}")
    plt.legend()
    plt.savefig(f"{plot_dir}/{config_name}_calls_evolution.png")
    plt.close()
    gc.collect()
    
    # 3. Unmet needs
    plt.figure(figsize=(12, 8))
    plt.plot(df['tick'], df['unmet_needs'], marker='')
    plt.title(f"Unmet Needs Evolution - {config_name}")
    plt.xlabel("Tick")
    plt.ylabel("Unassisted Cells (Level ≥ 4)")
    plt.savefig(f"{plot_dir}/{config_name}_unmet_needs.png")
    plt.close()
    gc.collect()
    
    # 4. Echo chamber effects
    plt.figure(figsize=(12, 8))
    plt.plot(df['tick'], df['human_echo_chamber'], label="Human Echo Chamber")
    plt.plot(df['tick'], df['human_ai_echo_chamber'], label="Human-AI Echo Chamber")
    plt.title(f"Echo Chamber Effects Over Time - {config_name}")
    plt.xlabel("Tick")
    plt.ylabel("Belief Variance (Higher = Less Echo Chamber)")
    plt.legend()
    plt.savefig(f"{plot_dir}/{config_name}_echo_chambers.png")
    plt.close()
    gc.collect()

def run_key_time_series_experiments():
    """Run a set of time series experiments for key configurations"""
    # Fixed parameters
    base_params = {
        'initial_trust': 0.5,
        'number_of_humans': 50,
        'share_confirming': 0.5,
        'shock_probability': 0.1,
        'shock_magnitude': 2,
        'trust_update_mode': "average",
        'exploitative_correction_factor': 1.0,
        'width': 50, 
        'height': 50,
        'ticks': 600
    }
    
    # Define key scenarios to run
    scenarios = {
        "baseline": {
            'share_exploitative': 0.5,
            'share_of_disaster': 0.2,
            'disaster_dynamics': 2,
            'ai_alignment_level': 0.5,
            'initial_ai_trust': 0.7
        },
        "high_exploratory": {
            'share_exploitative': 0.2,
            'share_of_disaster': 0.2,
            'disaster_dynamics': 2,
            'ai_alignment_level': 0.5,
            'initial_ai_trust': 0.7
        },
        "high_exploitative": {
            'share_exploitative': 0.8,
            'share_of_disaster': 0.2,
            'disaster_dynamics': 2,
            'ai_alignment_level': 0.5,
            'initial_ai_trust': 0.7
        },
        "low_ai_alignment": {
            'share_exploitative': 0.5,
            'share_of_disaster': 0.2,
            'disaster_dynamics': 2,
            'ai_alignment_level': 0.0,
            'initial_ai_trust': 0.7
        },
        "high_ai_alignment": {
            'share_exploitative': 0.5,
            'share_of_disaster': 0.2,
            'disaster_dynamics': 2,
            'ai_alignment_level': 1.0,
            'initial_ai_trust': 0.7
        },
        "high_disaster_dynamics": {
            'share_exploitative': 0.5,
            'share_of_disaster': 0.2,
            'disaster_dynamics': 3,
            'ai_alignment_level': 0.5,
            'initial_ai_trust': 0.7
        },
        "large_disaster": {
            'share_exploitative': 0.5,
            'share_of_disaster': 0.3,
            'disaster_dynamics': 2,
            'ai_alignment_level': 0.5,
            'initial_ai_trust': 0.7
        }
    }
    
    # Run each scenario one at a time to preserve memory
    for name, scenario_params in scenarios.items():
        print(f"Running time series experiment: {name}")
        params = {**base_params, **scenario_params}
        run_time_series_experiment(name, params)
        gc.collect()
    
    # Create comparative plots by loading data from files
    create_comparative_plots()

def create_comparative_plots():
    """Create plots comparing different scenarios by loading data from files"""
    plot_dir = "simulation_outputs/time_series/comparative"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get scenario names from saved files
    time_series_dir = "simulation_outputs/time_series"
    scenario_files = [f for f in os.listdir(time_series_dir) if f.endswith(".csv")]
    scenario_names = [os.path.splitext(f)[0] for f in scenario_files]
    
    # Create comparison plots for different metrics
    metrics_to_compare = [
        ('exp_ai_trust', "Trust in AI by Exploitative Agents"),
        ('expl_ai_trust', "Trust in AI by Exploratory Agents"),
        ('unmet_needs', "Unmet Needs"),
        ('human_echo_chamber', "Human Echo Chamber Effects"),
        ('human_ai_echo_chamber', "Human-AI Echo Chamber Effects")
    ]
    
    for metric, title in metrics_to_compare:
        plt.figure(figsize=(12, 8))
        
        for name in scenario_names:
            # Load data for this scenario
            df = pd.read_csv(f"{time_series_dir}/{name}.csv")
            plt.plot(df['tick'], df[metric], label=name)
            
            # Free memory
            del df
            gc.collect()
        
        plt.xlabel("Tick")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f"{title} Across Scenarios")
        plt.legend()
        plt.savefig(f"{plot_dir}/comparative_{metric}.png")
        plt.close()
        gc.collect()

if __name__ == "__main__":
    # Run parameter sweep experiments
    print("Running parameter sweep experiments...")
    run_experiments()
    
    # Run time series experiments
    print("Running time series experiments for key configurations...")
    run_key_time_series_experiments()
    
    print("All experiments completed. Results saved in simulation_outputs/")
