"""
Visualization utilities for Stackelberg optimization.

This module provides comprehensive visualization of optimization
progress, dependency graphs, and evolution trees.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.module import ModuleType
from ..core.candidate import SystemCandidate

logger = logging.getLogger(__name__)


class OptimizationVisualizer:
    """
    Comprehensive visualization for optimization progress.
    
    Creates various plots to visualize optimization progress, population
    evolution, module dependencies, and performance metrics.
    
    Attributes:
        save_dir: Directory for saving plots
        
    Examples:
        >>> visualizer = OptimizationVisualizer(save_dir=Path("plots"))
        >>> visualizer.plot_optimization_progress(optimizer, save=True)
        >>> visualizer.plot_evolution_tree(population, save=True)
    """
    
    def __init__(self, save_dir: Path = Path("optimization_plots")):
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_optimization_progress(self, optimizer: Any, save: bool = True):
        """
        Plot comprehensive optimization progress.
        
        Args:
            optimizer: Optimizer instance with population_manager
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Stackelberg Optimization Progress', fontsize=16)
        
        # Get data
        if hasattr(optimizer, 'population_manager'):
            pm = optimizer.population_manager
            generations = sorted(pm.generation_stats.keys())
        else:
            logger.warning("No population manager found")
            return
            
        if not generations:
            logger.warning("No generation data available")
            return
        
        stats = pm.generation_stats
        
        # Plot 1: Fitness over generations
        ax = axes[0, 0]
        avg_fitness = [stats[g]['avg_fitness'] for g in generations]
        best_fitness = [stats[g]['best_fitness'] for g in generations]
        
        ax.plot(generations, avg_fitness, 'b-', label='Average', linewidth=2)
        ax.plot(generations, best_fitness, 'r-', label='Best', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Diversity over generations
        ax = axes[0, 1]
        diversity = [stats[g]['diversity'] for g in generations]
        
        ax.plot(generations, diversity, 'g-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity')
        ax.set_title('Population Diversity')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Equilibrium values
        ax = axes[0, 2]
        if hasattr(optimizer, 'population') and optimizer.population:
            equilibrium_values = [c.equilibrium_value for c in optimizer.population]
            stability_scores = [c.stability_score for c in optimizer.population]
            
            ax.scatter(equilibrium_values, stability_scores, alpha=0.6, s=100)
            ax.set_xlabel('Equilibrium Value')
            ax.set_ylabel('Stability Score')
            ax.set_title('Strategic Distribution')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Module performance breakdown
        ax = axes[1, 0]
        module_scores = defaultdict(list)
        
        if hasattr(optimizer, 'population'):
            for candidate in optimizer.population:
                for trace in candidate.traces.values():
                    for module_name, score in trace.intermediate_scores.items():
                        module_scores[module_name].append(score)
        
        if module_scores:
            positions = range(len(module_scores))
            names = list(module_scores.keys())
            means = [np.mean(scores) for scores in module_scores.values()]
            stds = [np.std(scores) for scores in module_scores.values()]
            
            ax.bar(positions, means, yerr=stds, capsize=5)
            ax.set_xticks(positions)
            ax.set_xticklabels(names, rotation=45)
            ax.set_ylabel('Score')
            ax.set_title('Module Performance')
        
        # Plot 5: Innovation rate
        ax = axes[1, 1]
        innovation_counts = [stats[g]['innovations'] for g in generations]
        
        ax.plot(generations, innovation_counts, 'b-', label='Innovations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Innovations')
        ax.set_title('Innovation Rate')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Archive sizes
        ax = axes[1, 2]
        archive_data = {
            'Elite': len(pm.elite_archive),
            'Diversity': len(pm.diversity_archive),
            'Innovation': len(pm.innovation_archive)
        }
        
        ax.bar(archive_data.keys(), archive_data.values())
        ax.set_ylabel('Size')
        ax.set_title('Archive Sizes')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'optimization_progress.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimization progress plot to {self.save_dir}")
        else:
            plt.show()
    
    def plot_evolution_tree(self, population: List[SystemCandidate], save: bool = True):
        """
        Plot evolutionary tree of candidates.
        
        Args:
            population: List of candidates
            save: Whether to save the plot
        """
        # Build graph
        G = nx.DiGraph()
        
        # Add nodes
        for candidate in population:
            score = np.mean(list(candidate.scores.values())) if candidate.scores else 0
            G.add_node(candidate.candidate_id, 
                      score=score,
                      generation=candidate.generation)
        
        # Add edges
        for candidate in population:
            if candidate.parent_id is not None and candidate.parent_id in G:
                G.add_edge(candidate.parent_id, candidate.candidate_id)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_colors = [G.nodes[node]['score'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             cmap='viridis', node_size=300, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True)
        
        # Labels
        labels = {node: f"{node}\n{G.nodes[node]['score']:.2f}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Candidate Evolution Tree')
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                    label='Fitness Score')
        plt.axis('off')
        
        if save:
            plt.savefig(self.save_dir / 'evolution_tree.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved evolution tree to {self.save_dir}")
        else:
            plt.show()
    
    def plot_module_dependency_graph(self, dependency_analysis: Dict[str, Any],
                                   save: bool = True):
        """
        Plot module dependency graph.
        
        Args:
            dependency_analysis: Output from DependencyAnalyzer
            save: Whether to save the plot
        """
        G = dependency_analysis['graph']
        
        plt.figure(figsize=(10, 8))
        
        # Layout based on hierarchy
        if dependency_analysis['properties']['is_dag']:
            # Use hierarchical layout for DAG
            try:
                pos = nx.multipartite_layout(G, 
                                           subset_key=lambda n: len(nx.ancestors(G, n)))
            except:
                pos = nx.spring_layout(G, k=2)
        else:
            pos = nx.spring_layout(G, k=2)
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            if node in dependency_analysis['module_roles']:
                role = dependency_analysis['module_roles'][node]
                if role == ModuleType.LEADER:
                    node_colors.append('red')
                elif role == ModuleType.FOLLOWER:
                    node_colors.append('blue')
                else:
                    node_colors.append('green')
            else:
                node_colors.append('gray')
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=500, alpha=0.8)
        
        # Draw edges with different styles
        edge_styles = []
        for u, v in G.edges():
            edge_data = G.get_edge_data(u, v)
            if edge_data.get('type') == 'explicit':
                edge_styles.append('solid')
            else:
                edge_styles.append('dashed')
        
        # Draw edges (networkx doesn't support per-edge styles easily, so we group them)
        solid_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) 
                       if style == 'solid']
        dashed_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) 
                        if style == 'dashed']
        
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                             edge_color='gray', alpha=0.6, arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, 
                             edge_color='gray', alpha=0.6, arrows=True, style='dashed')
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=10, label='Leader'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='blue', markersize=10, label='Follower'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='green', markersize=10, label='Independent'),
            plt.Line2D([0], [0], color='gray', linewidth=2, 
                      linestyle='-', label='Explicit'),
            plt.Line2D([0], [0], color='gray', linewidth=2, 
                      linestyle='--', label='Implicit')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.title('Module Dependency Graph')
        plt.axis('off')
        
        if save:
            plt.savefig(self.save_dir / 'dependency_graph.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved dependency graph to {self.save_dir}")
        else:
            plt.show()
    
    def plot_performance_heatmap(self, population: List[SystemCandidate], 
                               save: bool = True):
        """
        Plot heatmap of module performance across population.
        
        Args:
            population: List of candidates
            save: Whether to save the plot
        """
        # Collect data
        module_names = set()
        for candidate in population:
            module_names.update(candidate.modules.keys())
        
        module_names = sorted(list(module_names))
        
        # Build performance matrix
        performance_matrix = []
        candidate_ids = []
        
        for candidate in population:
            row = []
            candidate_ids.append(f"C{candidate.candidate_id}")
            
            for module_name in module_names:
                # Get average score for this module
                scores = []
                for trace in candidate.traces.values():
                    if module_name in trace.intermediate_scores:
                        scores.append(trace.intermediate_scores[module_name])
                
                avg_score = np.mean(scores) if scores else 0.0
                row.append(avg_score)
            
            performance_matrix.append(row)
        
        if not performance_matrix:
            logger.warning("No performance data to plot")
            return
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        df = pd.DataFrame(performance_matrix, 
                         index=candidate_ids, 
                         columns=module_names)
        
        sns.heatmap(df, cmap='RdYlGn', center=0.5, 
                   annot=True, fmt='.2f', cbar_kws={'label': 'Score'})
        
        plt.title('Module Performance Heatmap')
        plt.xlabel('Module')
        plt.ylabel('Candidate')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'performance_heatmap.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance heatmap to {self.save_dir}")
        else:
            plt.show()
    
    def plot_pareto_front(self, population: List[SystemCandidate], 
                         save: bool = True):
        """
        Plot Pareto front for multi-objective optimization.
        
        Args:
            population: List of candidates
            save: Whether to save the plot
        """
        # Extract objectives
        performances = []
        equilibriums = []
        stabilities = []
        
        for candidate in population:
            if candidate.scores:
                performances.append(np.mean(list(candidate.scores.values())))
                equilibriums.append(candidate.equilibrium_value)
                stabilities.append(candidate.stability_score)
        
        if not performances:
            logger.warning("No data for Pareto front")
            return
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(performances, equilibriums, stabilities, 
                           c=performances, cmap='viridis', s=100, alpha=0.6)
        
        ax.set_xlabel('Performance')
        ax.set_ylabel('Equilibrium Value')
        ax.set_zlabel('Stability Score')
        ax.set_title('Multi-Objective Pareto Front')
        
        plt.colorbar(scatter, label='Performance')
        
        if save:
            plt.savefig(self.save_dir / 'pareto_front.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved Pareto front to {self.save_dir}")
        else:
            plt.show()