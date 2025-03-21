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
from copy import deepcopy

# Import your model classes
from DisasterModelNew import DisasterModel, HumanAgent, AIAgent

# Ensure output directory exists
os.makedirs("simulation_outputs", exist_ok=True)

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
    
    # Run experiments in parallel
    num_processes = min(mp.cpu_count(), len(param_combinations))
    print(f"Running {len(param_combinations)} experiments with {num_processes} parallel processes")
    
    # Split combinations for parallel processing
    chunks = np.array_split(param_combinations, num_processes)
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, [(chunk, fixed_params, experiment_params) for chunk in chunks])
    
    # Combine results from all processes
    all_results = []
    for r in results:
        all_results.extend(r)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Save the raw results
    results_df.to_csv("simulation_outputs/experiment_results.csv", index=False)
    
    # Generate analysis and visualizations
    analyze_results(results_df, experiment_params)

def process_chunk(args):
    chunk, fixed_params, param_names = args
    results = []
    
    for params in chunk:
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
        results.append(result)
    
    return results

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
    
    # 1. Impact of share_exploitative on key metrics
    explore_plot_key_metrics(results_df, 'share_exploitative', 
                        ['human_echo_chamber', 'assistance_coverage', 'incorrect_assistance_rate', 'avg_unmet_needs'],
                        "Impact of Exploitative/Exploratory Ratio",
                        plot_dir)
    
    # 2. Impact of ai_alignment_level on key metrics
    explore_plot_key_metrics(results_df, 'ai_alignment_level', 
                        ['human_ai_echo_chamber', 'assistance_coverage', 'avg_exp_ai_trust', 'avg_expl_ai_trust'],
                        "Impact of AI Alignment Level",
                        plot_dir)
    
    # 3. Impact of disaster_dynamics on key metrics
    explore_plot_key_metrics(results_df, 'disaster_dynamics', 
                        ['assistance_coverage', 'incorrect_assistance_rate', 'avg_unmet_needs'],
                        "Impact of Disaster Dynamics",
                        plot_dir)
    
    # 4. Impact of share_of_disaster on key metrics
    explore_plot_key_metrics(results_df, 'share_of_disaster', 
                        ['assistance_coverage', 'incorrect_assistance_rate', 'avg_unmet_needs'],
                        "Impact of Disaster Coverage",
                        plot_dir)
    
    # 5. Compare human vs AI trust across agent types (exploitative vs exploratory)
    plot_trust_comparison(results_df, plot_dir)
    
    # 6. Heatmap of key interactions (e.g., ai_alignment_level vs share_exploitative)
    for metric in ['assistance_coverage', 'human_ai_echo_chamber']:
        plot_interaction_heatmap(results_df, 'ai_alignment_level', 'share_exploitative', 
                              metric, plot_dir, f"{metric}_heatmap")
    
    # 7. Plot call ratios (human vs AI) for different agent types
    plot_call_ratios(results_df, plot_dir)

def explore_plot_key_metrics(df, param_name, metrics, title, plot_dir):
    """Create box plots for key metrics by parameter variations"""
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=param_name, y=metric, data=df)
        plt.title(f"{title} on {metric}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{param_name}_{metric}_boxplot.png")
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

# Function to run a time-series experiment for a specific configuration
def run_time_series_experiment(config_name, params):
    """Run a single experiment and save time series data"""
    # Extract parameters for this run
    ticks = params.pop('ticks')  # Remove ticks from params for model init
    ai_alignment_level = params.pop('ai_alignment_level')  # Remove for model init
    
    # Create model instance
    model = DisasterModel(**params)
    model.ai_alignment_level = ai_alignment_level
    
    # Prepare data structures for time series
    time_series = {
        'tick': [],
        'exp_human_trust': [],
        'exp_ai_trust': [],
        'expl_human_trust': [],
        'expl_ai_trust': [],
        'calls_exp_human': [],
        'calls_exp_ai': [],
        'calls_expl_human': [],
        'calls_expl_ai': [],
        'unmet_needs': [],
        'human_echo_chamber': [],
        'human_ai_echo_chamber': []
    }
    
    # Run simulation and collect time series data
    for i in range(ticks):
        model.step()
        
        # Record data at each tick
        time_series['tick'].append(i)
        time_series['exp_human_trust'].append(model.trust_data[-1][0])
        time_series['exp_ai_trust'].append(model.trust_data[-1][1])
        time_series['expl_human_trust'].append(model.trust_data[-1][2])
        time_series['expl_ai_trust'].append(model.trust_data[-1][3])
        time_series['calls_exp_human'].append(model.calls_data[-1][0])
        time_series['calls_exp_ai'].append(model.calls_data[-1][1])
        time_series['calls_expl_human'].append(model.calls_data[-1][2])
        time_series['calls_expl_ai'].append(model.calls_data[-1][3])
        time_series['unmet_needs'].append(model.unmet_needs_evolution[-1])
        time_series['human_echo_chamber'].append(calculate_belief_variance(model, "human_only"))
        time_series['human_ai_echo_chamber'].append(calculate_belief_variance(model, "human_ai"))
    
    # Convert to DataFrame
    df = pd.DataFrame(time_series)
    
    # Save to CSV
    os.makedirs("simulation_outputs/time_series", exist_ok=True)
    df.to_csv(f"simulation_outputs/time_series/{config_name}.csv", index=False)
    
    # Create key plots
    plot_time_series(df, config_name)
    
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
    
    # 3. Unmet needs
    plt.figure(figsize=(12, 8))
    plt.plot(df['tick'], df['unmet_needs'], marker='')
    plt.title(f"Unmet Needs Evolution - {config_name}")
    plt.xlabel("Tick")
    plt.ylabel("Unassisted Cells (Level ≥ 4)")
    plt.savefig(f"{plot_dir}/{config_name}_unmet_needs.png")
    plt.close()
    
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
    
    # Run each scenario and collect results
    time_series_results = {}
    for name, scenario_params in scenarios.items():
        print(f"Running time series experiment: {name}")
        params = {**base_params, **scenario_params}
        time_series_results[name] = run_time_series_experiment(name, params)
    
    # Create comparative plots
    create_comparative_plots(time_series_results)

def create_comparative_plots(results_dict):
    """Create plots comparing different scenarios"""
    plot_dir = "simulation_outputs/time_series/comparative"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Compare trust in AI across scenarios
    plt.figure(figsize=(12, 8))
    for name, df in results_dict.items():
        plt.plot(df['tick'], df['exp_ai_trust'], label=f"{name} - Exploitative")
    plt.xlabel("Tick")
    plt.ylabel("Average Trust in AI")
    plt.title("Trust in AI by Exploitative Agents Across Scenarios")
    plt.legend()
    plt.savefig(f"{plot_dir}/comparative_exp_ai_trust.png")
    plt.close()
    
    # Compare trust in AI by exploratory agents
    plt.figure(figsize=(12, 8))
    for name, df in results_dict.items():
        plt.plot(df['tick'], df['expl_ai_trust'], label=f"{name} - Exploratory")
    plt.xlabel("Tick")
    plt.ylabel("Average Trust in AI")
    plt.title("Trust in AI by Exploratory Agents Across Scenarios")
    plt.legend()
    plt.savefig(f"{plot_dir}/comparative_expl_ai_trust.png")
    plt.close()
    
    # Compare unmet needs
    plt.figure(figsize=(12, 8))
    for name, df in results_dict.items():
        plt.plot(df['tick'], df['unmet_needs'], label=name)
    plt.xlabel("Tick")
    plt.ylabel("Unmet Needs")
    plt.title("Unmet Needs Across Scenarios")
    plt.legend()
    plt.savefig(f"{plot_dir}/comparative_unmet_needs.png")
    plt.close()
    
    # Compare human echo chamber effects
    plt.figure(figsize=(12, 8))
    for name, df in results_dict.items():
        plt.plot(df['tick'], df['human_echo_chamber'], label=name)
    plt.xlabel("Tick")
    plt.ylabel("Belief Variance")
    plt.title("Human Echo Chamber Effects Across Scenarios")
    plt.legend()
    plt.savefig(f"{plot_dir}/comparative_human_echo_chamber.png")
    plt.close()
    
    # Compare human-AI echo chamber effects
    plt.figure(figsize=(12, 8))
    for name, df in results_dict.items():
        plt.plot(df['tick'], df['human_ai_echo_chamber'], label=name)
    plt.xlabel("Tick")
    plt.ylabel("Belief Variance")
    plt.title("Human-AI Echo Chamber Effects Across Scenarios")
    plt.legend()
    plt.savefig(f"{plot_dir}/comparative_human_ai_echo_chamber.png")
    plt.close()

if __name__ == "__main__":
    # Run parameter sweep experiments
    print("Running parameter sweep experiments...")
    run_experiments()
    
    # Run time series experiments
    print("Running time series experiments for key configurations...")
    run_key_time_series_experiments()
    
    print("All experiments completed. Results saved in simulation_outputs/")
