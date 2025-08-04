"""
Dependency analysis for compound AI systems.

This module analyzes dependencies between modules using graph
algorithms and data flow analysis.
"""

import json
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Any, Set, Optional

import networkx as nx
import numpy as np

from ..core.module import Module, ModuleType
from ..core.candidate import ExecutionTrace

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """
    Comprehensive dependency analysis with graph algorithms.
    
    Analyzes both explicit dependencies (from module definitions) and
    implicit dependencies (from execution patterns and data flow).
    
    Examples:
        >>> analyzer = DependencyAnalyzer()
        >>> analysis = analyzer.analyze_dependencies(
        ...     modules, traces, prompts_only=False
        ... )
    """
    
    def __init__(self):
        self.dependency_cache = {}
    
    def analyze_dependencies(
        self, 
        modules: Dict[str, Module], 
        traces: Optional[List[ExecutionTrace]] = None,
        prompts_only: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive dependency analysis.
        
        Args:
            modules: Dictionary of modules to analyze
            traces: Optional execution traces for deeper analysis
            prompts_only: If True, only analyze explicit dependencies
            
        Returns:
            Dictionary containing dependency analysis results
        """
        # Build dependency graph
        dep_graph = nx.DiGraph()
        
        # Add nodes
        for name, module in modules.items():
            dep_graph.add_node(name, module=module)
        
        # Add explicit dependencies
        for name, module in modules.items():
            for dep in module.dependencies:
                if dep in modules:
                    dep_graph.add_edge(dep, name, type='explicit', weight=1.0)
        
        if not prompts_only and traces:
            # Add implicit dependencies from traces
            implicit_deps = self._infer_implicit_dependencies(traces)
            for target, sources in implicit_deps.items():
                for source in sources:
                    if not dep_graph.has_edge(source, target):
                        dep_graph.add_edge(source, target, type='implicit', weight=0.8)
            
            # Add data flow dependencies
            flow_deps = self._analyze_data_flow(traces)
            for source, targets in flow_deps.items():
                for target, weight in targets.items():
                    if not dep_graph.has_edge(source, target):
                        dep_graph.add_edge(source, target, type='data_flow', weight=weight)
        
        # Analyze graph properties
        analysis = {
            'dependencies': {name: list(dep_graph.predecessors(name)) for name in modules},
            'graph': dep_graph,
            'properties': self._analyze_graph_properties(dep_graph),
            'module_roles': self._determine_module_roles(dep_graph),
            'critical_paths': self._find_critical_paths(dep_graph),
            'clusters': self._find_module_clusters(dep_graph)
        }
        
        return analysis
    
    def _infer_implicit_dependencies(self, traces: List[ExecutionTrace]) -> Dict[str, Set[str]]:
        """Infer dependencies from execution patterns."""
        implicit_deps = defaultdict(set)
        
        # Analyze execution sequences
        execution_sequences = defaultdict(lambda: defaultdict(int))
        
        for trace in traces:
            if len(trace.execution_order) > 1:
                for i in range(len(trace.execution_order) - 1):
                    current = trace.execution_order[i]
                    next_module = trace.execution_order[i + 1]
                    execution_sequences[next_module][current] += 1
        
        # Find consistent patterns
        for target, sources in execution_sequences.items():
            total_executions = sum(sources.values())
            for source, count in sources.items():
                if count / total_executions > 0.8:  # 80% consistency threshold
                    implicit_deps[target].add(source)
        
        # Analyze output-input relationships
        for trace in traces:
            for i, module in enumerate(trace.execution_order):
                if module in trace.module_outputs:
                    output = str(trace.module_outputs[module])
                    
                    # Check subsequent modules
                    for j in range(i + 1, len(trace.execution_order)):
                        next_module = trace.execution_order[j]
                        if next_module in trace.module_inputs:
                            next_input = str(trace.module_inputs[next_module])
                            
                            # Check for data flow
                            if self._check_data_flow(output, next_input):
                                implicit_deps[next_module].add(module)
        
        return implicit_deps
    
    def _analyze_data_flow(self, traces: List[ExecutionTrace]) -> Dict[str, Dict[str, float]]:
        """Analyze data flow between modules."""
        data_flows = defaultdict(lambda: defaultdict(float))
        
        for trace in traces:
            # Use causal links if available
            for source, targets in trace.causal_links.items():
                for target in targets:
                    data_flows[source][target] += 1.0
            
            # Also analyze based on content similarity
            modules_with_output = [
                (i, m) for i, m in enumerate(trace.execution_order) 
                if m in trace.module_outputs
            ]
            
            for i, (idx1, module1) in enumerate(modules_with_output):
                output1 = trace.module_outputs[module1]
                
                for j in range(i + 1, len(modules_with_output)):
                    idx2, module2 = modules_with_output[j]
                    
                    if module2 in trace.module_inputs:
                        input2 = trace.module_inputs[module2]
                        
                        # Calculate flow strength
                        flow_strength = self._calculate_flow_strength(output1, input2)
                        if flow_strength > 0.3:
                            data_flows[module1][module2] += flow_strength
        
        # Normalize flows
        normalized_flows = defaultdict(dict)
        for source, targets in data_flows.items():
            total_flow = sum(targets.values())
            if total_flow > 0:
                for target, flow in targets.items():
                    normalized_flows[source][target] = flow / total_flow
        
        return normalized_flows
    
    def _check_data_flow(self, output: str, input: str) -> bool:
        """Check if output flows into input."""
        # Simple heuristic - check for substring match
        if len(output) > 20 and output[:50] in input:
            return True
        
        # Check for structural similarity (e.g., JSON keys)
        try:
            output_data = json.loads(output)
            input_data = json.loads(input)
            
            # Check for key overlap
            if isinstance(output_data, dict) and isinstance(input_data, dict):
                output_keys = set(output_data.keys())
                input_keys = set(input_data.keys())
                
                overlap = len(output_keys & input_keys) / len(output_keys) if output_keys else 0
                return overlap > 0.5
        except:
            pass
        
        return False
    
    def _calculate_flow_strength(self, output: Any, input: Any) -> float:
        """Calculate strength of data flow."""
        if isinstance(output, str) and isinstance(input, str):
            # String similarity
            return SequenceMatcher(None, output[:100], input[:100]).ratio()
        
        elif isinstance(output, dict) and isinstance(input, dict):
            # Key overlap for dictionaries
            output_keys = set(output.keys())
            input_keys = set(input.keys())
            
            if output_keys:
                return len(output_keys & input_keys) / len(output_keys)
        
        elif isinstance(output, list) and isinstance(input, list):
            # Element overlap for lists
            if output and input:
                output_set = set(str(x) for x in output[:10])
                input_set = set(str(x) for x in input[:10])
                
                if output_set:
                    return len(output_set & input_set) / len(output_set)
        
        return 0.0
    
    def _analyze_graph_properties(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze structural properties of dependency graph."""
        properties = {
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'connected_components': list(nx.weakly_connected_components(graph)),
            'strongly_connected': list(nx.strongly_connected_components(graph))
        }
        
        if properties['is_dag']:
            # DAG-specific properties
            properties['topological_order'] = list(nx.topological_sort(graph))
            properties['longest_path_length'] = nx.dag_longest_path_length(graph)
            
            # Find sources and sinks
            properties['sources'] = [n for n in graph.nodes() if graph.in_degree(n) == 0]
            properties['sinks'] = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        else:
            # Find cycles
            properties['cycles'] = list(nx.simple_cycles(graph))
        
        # Centrality measures
        properties['centrality'] = {
            'in_degree': nx.in_degree_centrality(graph),
            'out_degree': nx.out_degree_centrality(graph),
            'betweenness': nx.betweenness_centrality(graph)
        }
        
        return properties
    
    def _determine_module_roles(self, graph: nx.DiGraph) -> Dict[str, ModuleType]:
        """Determine module roles based on graph structure."""
        roles = {}
        
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            if in_degree == 0 and out_degree > 0:
                # No dependencies, influences others = Leader
                roles[node] = ModuleType.LEADER
            elif in_degree > 0 and out_degree == 0:
                # Has dependencies, doesn't influence others = Follower
                roles[node] = ModuleType.FOLLOWER
            elif in_degree > 0 and out_degree > 0:
                # Both depends and influences = Follower (intermediate)
                roles[node] = ModuleType.FOLLOWER
            else:
                # No dependencies, no influence = Independent
                roles[node] = ModuleType.INDEPENDENT
        
        # Refine based on centrality
        centrality = nx.betweenness_centrality(graph)
        
        for node, cent in centrality.items():
            if cent > 0.5 and roles[node] == ModuleType.FOLLOWER:
                # High centrality followers might act as sub-leaders
                # Keep as follower but note the importance
                if node not in graph.graph:
                    graph.nodes[node]['importance'] = 'high'
        
        return roles
    
    def _find_critical_paths(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find critical execution paths."""
        critical_paths = []
        
        if not nx.is_directed_acyclic_graph(graph):
            return critical_paths
        
        # Find all paths from sources to sinks
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        for source in sources:
            for sink in sinks:
                try:
                    # Find all simple paths
                    paths = list(nx.all_simple_paths(graph, source, sink))
                    
                    # Find longest/most weighted path
                    if paths:
                        # Sort by path weight (sum of edge weights)
                        weighted_paths = []
                        for path in paths:
                            weight = 0
                            for i in range(len(path) - 1):
                                edge_data = graph.get_edge_data(path[i], path[i+1])
                                weight += edge_data.get('weight', 1.0)
                            weighted_paths.append((path, weight))
                        
                        # Take the highest weighted path
                        best_path = max(weighted_paths, key=lambda x: x[1])
                        critical_paths.append(best_path[0])
                except nx.NetworkXNoPath:
                    continue
        
        # Remove duplicate paths
        unique_paths = []
        for path in critical_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        
        return unique_paths
    
    def _find_module_clusters(self, graph: nx.DiGraph) -> List[Set[str]]:
        """Find clusters of tightly connected modules."""
        # Convert to undirected for community detection
        undirected = graph.to_undirected()
        
        # Find communities
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(undirected))
            return [set(community) for community in communities]
        except:
            # Fallback to connected components
            return [set(comp) for comp in nx.connected_components(undirected)]