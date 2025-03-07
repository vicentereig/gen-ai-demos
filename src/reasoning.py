import dspy
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


# Define the signature for the initial thoughts generation
class InitialThoughtsGenerator(dspy.Signature):
    """Generate initial thoughts for decomposing a complex problem."""
    query = dspy.InputField(desc="The original query or problem to solve")
    max_thoughts = dspy.InputField(desc="Maximum number of initial thoughts to generate")

    thoughts = dspy.OutputField(
        desc="A list of complete, meaningful thoughts to tackle the problem. Each thought should be a fully formed idea, not individual characters or fragments.")
    strategy = dspy.OutputField(desc="A strategy to guide the exploration of these thoughts")


# Define the signature for generating subsequent thoughts
class ThoughtExpander(dspy.Signature):
    """Generate additional thoughts based on the current graph state."""
    query = dspy.InputField(desc="The original query or problem to solve")
    current_graph = dspy.InputField(desc="Current state of the graph including all thoughts and their answers")
    max_new_thoughts = dspy.InputField(desc="Maximum number of new thoughts to generate")

    new_thoughts = dspy.OutputField(
        desc="A list of new complete, meaningful thoughts to add to the graph. Each thought should be a fully formed idea, not individual characters or fragments.")
    new_edges = dspy.OutputField(desc="A list of edges to add, specified as [(source_id, target_id), ...] tuples")
    strategy = dspy.OutputField(desc="A strategy to guide the exploration of these thoughts")


# Define a signature for complexity checking
class ComplexityChecker(dspy.Signature):
    """Determine if a thought is complex enough to warrant further decomposition."""
    thought = dspy.InputField(desc="The thought content to evaluate")
    graph_context = dspy.InputField(desc="Current state of the graph for context")

    is_complex = dspy.OutputField(desc="1 if the thought is complex and should be decomposed, 0 otherwise")
    reason = dspy.OutputField(desc="Reasoning for the complexity determination")


# Define a signature for thought evaluation
class ThoughtEvaluator(dspy.Signature):
    """Evaluate a thought to produce an answer or response."""
    thought = dspy.InputField(desc="The thought to evaluate")
    graph_context = dspy.InputField(desc="Current graph context to inform the evaluation")

    answer = dspy.OutputField(desc="Answer or response to the thought")


# Define a signature for final answer synthesis
class FinalAnswerSynthesizer(dspy.Signature):
    """Synthesize a final answer from the complete graph of thoughts."""
    query = dspy.InputField(desc="The original query or problem")
    graph = dspy.InputField(desc="The complete graph with all thoughts and answers")

    final_answer = dspy.OutputField(desc="The final answer to the original query")


# Create DSPy modules for each signature
class T0(dspy.Module):
    """Module for generating initial thoughts (T₀ in the paper)."""

    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(InitialThoughtsGenerator)

    def forward(self, query, max_thoughts=3):
        # Ensure the query is treated as a complete prompt, not split
        return self.generator(query=query, max_thoughts=max_thoughts)


class Te(dspy.Module):
    """Module for expanding thoughts with new nodes and edges (Tₑ in the paper)."""

    def __init__(self):
        super().__init__()
        self.expander = dspy.ChainOfThought(ThoughtExpander)

    def forward(self, query, current_graph, max_new_thoughts=3):
        return self.expander(
            query=query,
            current_graph=current_graph,
            max_new_thoughts=max_new_thoughts
        )


class C(dspy.Module):
    """Module for complexity checking (C in the paper)."""

    def __init__(self):
        super().__init__()
        self.checker = dspy.Predict(ComplexityChecker)

    def forward(self, thought, graph_context):
        result = self.checker(thought=thought, graph_context=graph_context)
        return int(result.is_complex), result.reason


class Eval(dspy.Module):
    """Module for thought evaluation (Eval in the paper)."""

    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(ThoughtEvaluator)

    def forward(self, thought, graph_context):
        return self.evaluator(thought=thought, graph_context=graph_context).answer


class Phi(dspy.Module):
    """Module for final answer synthesis (Φ in the paper)."""

    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.Predict(FinalAnswerSynthesizer)

    def forward(self, query, graph):
        return self.synthesizer(query=query, graph=graph).final_answer


# Define a data structure for thoughts
@dataclass
class Thought:
    id: str
    content: str
    layer: int
    strategy: str
    answer: Optional[str] = None
    is_complex: bool = False
    nested_graph: Optional[Any] = None

    def __str__(self):
        status = "COMPLEX" if self.is_complex else "EVALUATED"
        if self.answer:
            status = f"{status}: {self.answer[:50]}..." if len(self.answer) > 50 else f"{status}: {self.answer}"
        return f"Thought {self.id} [Layer {self.layer}]: {self.content[:50]}... ({status})"


# The main AGoT implementation
class AdaptiveGraphOfThoughts(dspy.Module):
    def __init__(self, max_depth=1, max_layers=3, max_nodes=3, token_budget=float('inf')):
        super().__init__()

        # Initialize the DSPy modules
        self.t0 = T0()
        self.te = Te()
        self.c = C()
        self.eval = Eval()
        self.phi = Phi()

        # Set limits
        self.max_depth = max_depth
        self.max_layers = max_layers
        self.max_nodes = max_nodes
        self.token_budget = token_budget

    def _format_graph_for_context(self, G):
        """Format the graph state as a string for context in prompts."""
        result = []
        for node_id in nx.topological_sort(G):
            thought = G.nodes[node_id]['data']
            result.append(str(thought))

            # Add edge information
            for successor in G.successors(node_id):
                result.append(f"  ↳ connects to Thought {successor}")

        return "\n".join(result)

    def _is_edge_safe(self, graph, source, target):
        """
        Check if adding an edge would create a cycle in the graph.

        Args:
            G (nx.DiGraph): The current directed graph
            source (str): Source node ID
            target (str): Target node ID

        Returns:
            bool: True if adding the edge would not create a cycle, False otherwise
        """
        # Create a copy of the graph to test edge addition
        test_graph = graph.copy()
        test_graph.add_edge(source, target)

        try:
            # Check if there are any cycles in the graph
            nx.find_cycle(test_graph)
            return False  # Cycle found, edge is not safe
        except nx.NetworkXNoCycle:
            return True  # No cycle, edge i

    def _recursive_agot(self, query, heritage, parent_graph=None, depth=0):
        """Recursive implementation of the AGoT algorithm."""
        # Initialize graph for this recursion level
        G = nx.DiGraph()

        # Process layer by layer
        for layer in range(self.max_layers):
            # Generate thoughts for this layer
            if layer == 0 and depth == 0:
                # Generate initial thoughts for the top-level graph
                # Add explicit instructions to handle multilingual queries properly
                enhanced_query = f"""
                IMPORTANT: Please process this query in its original language.
                DO NOT split the query into individual characters.
                Generate {self.max_nodes} complete, meaningful thoughts related to this query.
                Query: {query}
                """

                print(f"\n[DEBUG] Depth {depth}, Layer {layer}: Generating initial thoughts for query: {query}")
                result = self.t0(enhanced_query, self.max_nodes)

                # Extract thoughts - ensure we have proper complete thoughts
                thoughts = result.thoughts
                if not thoughts or isinstance(thoughts, str):
                    # Fallback if the output format is incorrect
                    if isinstance(thoughts, str):
                        # Try to parse if it's a string
                        import re
                        thoughts = re.split(r'\d+\.\s+', thoughts)
                        thoughts = [t.strip() for t in thoughts if t.strip()]

                    # If still empty, create default thoughts
                    if not thoughts:
                        thoughts = [
                            f"Analyze key aspects of {query}",
                            f"Identify main components in {query}",
                            f"Determine methodological approach for {query}"
                        ]

                print(f"[DEBUG] Generated thoughts: {thoughts}")
                strategy = result.strategy

                # Add initial thoughts to the graph
                for i, thought_content in enumerate(thoughts):
                    # Skip if the thought is just a single character or empty
                    if len(thought_content.strip()) <= 1:
                        print(f"[DEBUG] Skipping single character thought: '{thought_content}'")
                        continue

                    thought_id = f"{layer}_{i}"
                    G.add_node(thought_id, data=Thought(
                        id=thought_id,
                        content=thought_content,
                        layer=layer,
                        strategy=strategy
                    ))
                    print(f"[DEBUG] Added thought {thought_id}: {thought_content[:50]}...")

            elif layer == 0:
                # Generate initial thoughts for a nested graph
                parent_context = self._format_graph_for_context(parent_graph)

                enhanced_query = f"""
                IMPORTANT: Please process this query in its original language.
                DO NOT split the query into individual characters.
                Generate {self.max_nodes} complete, meaningful thoughts related to this query.
                Query: {query}
                Context: {parent_context}
                """

                print(f"\n[DEBUG] Depth {depth}, Layer {layer}: Generating nested initial thoughts for query: {query}")
                result = self.t0(enhanced_query, self.max_nodes)

                # Extract and validate thoughts
                thoughts = result.thoughts
                if not thoughts or isinstance(thoughts, str):
                    if isinstance(thoughts, str):
                        import re
                        thoughts = re.split(r'\d+\.\s+', thoughts)
                        thoughts = [t.strip() for t in thoughts if t.strip()]

                    if not thoughts:
                        thoughts = [
                            f"Decompose {query} into smaller components",
                            f"Identify key challenges in {query}",
                            f"Outline approach for addressing {query}"
                        ]

                print(f"[DEBUG] Generated nested thoughts: {thoughts}")
                strategy = result.strategy

                # Add initial thoughts to the graph
                for i, thought_content in enumerate(thoughts):
                    # Skip if the thought is too short
                    if len(thought_content.strip()) <= 1:
                        print(f"[DEBUG] Skipping single character thought: '{thought_content}'")
                        continue

                    thought_id = f"{layer}_{i}"
                    G.add_node(thought_id, data=Thought(
                        id=thought_id,
                        content=thought_content,
                        layer=layer,
                        strategy=strategy
                    ))
                    print(f"[DEBUG] Added nested thought {thought_id}: {thought_content[:50]}...")
            else:
                # Generate subsequent thoughts
                graph_context = self._format_graph_for_context(G)

                enhanced_query = f"""
                IMPORTANT: Please process this query in its original language.
                DO NOT split the query into individual characters.
                Generate {self.max_nodes} complete, meaningful thoughts that build on existing thoughts.

                Query: {query}

                Current thoughts: {graph_context}

                When providing edges, please use this format exactly: [(source_id, target_id), (source_id, target_id)]
                For example: [("0_0", "1_0"), ("0_1", "1_1")]
                """

                print(f"\n[DEBUG] Depth {depth}, Layer {layer}: Generating subsequent thoughts")
                result = self.te(enhanced_query, graph_context, self.max_nodes)

                # Extract and validate new thoughts
                new_thoughts = result.new_thoughts
                new_edges = result.new_edges
                if not new_thoughts or isinstance(new_thoughts, str):
                    # Fallback if the output format is incorrect
                    if isinstance(new_thoughts, str):
                        # Try to parse if it's a string
                        import re
                        new_thoughts = re.split(r'\d+\.\s+', new_thoughts)
                        new_thoughts = [t.strip() for t in new_thoughts if t.strip()]

                    # If still empty, create default thoughts based on existing nodes
                    if not new_thoughts:
                        prev_thoughts = [data['data'].content for _, data in G.nodes(data=True)]
                        new_thoughts = [
                            f"Explore implications of {prev_thoughts[0] if prev_thoughts else query}",
                            f"Consider alternative perspectives on {prev_thoughts[-1] if prev_thoughts else query}",
                            f"Synthesize findings from previous thoughts"
                        ]

                print(f"[DEBUG] Generated subsequent thoughts: {new_thoughts}")
                print(f"[DEBUG] New edges: {new_edges}")

                # Ensure new_edges is in the correct format
                if new_edges:
                    # If new_edges is a string, try to parse it
                    if isinstance(new_edges, str):
                        try:
                            import ast
                            parsed_edges = ast.literal_eval(new_edges)
                            if isinstance(parsed_edges, list):
                                new_edges = parsed_edges
                            else:
                                new_edges = []
                        except:
                            # If parsing fails, set to empty list
                            print("[DEBUG] Error parsing edges, using empty list")
                            new_edges = []
                else:
                    new_edges = []

                # Make sure new_edges is a list of tuples
                if not all(isinstance(edge, tuple) and len(edge) == 2 for edge in new_edges):
                    print("[DEBUG] Edges not in correct format, creating default edges")
                    # Create default edges connecting previous layer to this one
                    new_edges = []
                    prev_layer_nodes = [n for n, data in G.nodes(data=True)
                                        if data['data'].layer == layer - 1]

                    for prev_node in prev_layer_nodes:
                        for i in range(len(new_thoughts)):
                            if len(new_thoughts[i].strip()) > 1:  # Skip single character thoughts
                                new_edges.append((prev_node, f"{layer}_{i}"))

                strategy = result.strategy

                # Add new thoughts to the graph
                for i, thought_content in enumerate(new_thoughts):
                    # Skip if the thought is just a single character or empty
                    if len(thought_content.strip()) <= 1:
                        print(f"[DEBUG] Skipping single character thought: '{thought_content}'")
                        continue

                    thought_id = f"{layer}_{i}"
                    G.add_node(thought_id, data=Thought(
                        id=thought_id,
                        content=thought_content,
                        layer=layer,
                        strategy=strategy
                    ))
                    print(f"[DEBUG] Added subsequent thought {thought_id}: {thought_content[:50]}...")

                # Add edges
                print(f"[DEBUG] Adding edges: {new_edges}")
                for edge in new_edges:
                    try:
                        source, target = edge
                        if source in G and target in G:
                            if self._is_edge_safe(G, source, target):
                                print(f"[DEBUG] Adding edge... {source} → {target}")
                                G.add_edge(source, target)
                                print(f"[DEBUG] Added edge: {source} → {target}")
                            else:
                                print(f"[DEBUG] Edge not added, cycle found: {new_edges}")
                        else:
                            print(f"[DEBUG] Edge not added, nodes do not exist: {source} → {target}")
                    except ValueError as e:
                        print(f"[DEBUG] Error unpacking edge {edge}: {e}")
                        # Continue with other edges

            # Check if any thought in this layer is final
            for node_id, data in list(G.nodes(data=True)):
                thought = data['data']
                if thought.layer == layer:
                    # Check if this thought is complex
                    graph_context = self._format_graph_for_context(G)

                    print(
                        f"[DEBUG] Depth {depth}, Layer {layer}: Checking complexity of thought {node_id}: {thought.content[:50]}...")
                    is_complex, reason = self.c(thought.content, graph_context)
                    print(f"[DEBUG] Complexity check result: {is_complex}, Reason: {reason}")

                    if is_complex and depth < self.max_depth:
                        # Mark as complex
                        thought.is_complex = True
                        print(f"[DEBUG] Thought {node_id} is complex, starting recursive processing")

                        # Recursively process this thought
                        nested_heritage = heritage + [(layer, int(node_id.split('_')[1]))]
                        answer, nested_graph = self._recursive_agot(
                            thought.content,
                            nested_heritage,
                            G,
                            depth + 1
                        )

                        # Store the result
                        thought.answer = answer
                        thought.nested_graph = nested_graph
                        print(f"[DEBUG] Completed recursive processing for thought {node_id}, answer: {answer[:50]}...")

                    else:
                        # Evaluate the thought directly
                        thought.is_complex = False
                        print(f"[DEBUG] Evaluating thought {node_id} directly")
                        thought.answer = self.eval(thought.content, graph_context)
                        print(f"[DEBUG] Thought {node_id} evaluation result: {thought.answer[:50]}...")

            # Check if we need to add another layer
            if layer == self.max_layers - 1 or self._should_terminate(G):
                break

        # Synthesize final answer from this graph
        graph_context = self._format_graph_for_context(G)
        print(f"\n[DEBUG] Depth {depth}: Synthesizing final answer")
        final_answer = self.phi(query, graph_context)
        print(f"[DEBUG] Final answer: {final_answer[:100]}...")

        return final_answer, G

    def _should_terminate(self, G):
        """Check if we should terminate early based on the graph state.

        Terminate if:
          - Any thought in the graph has produced an answer (i.e. a non-empty answer exists).
          - The estimated token usage exceeds the token budget.
        """
        # Check if any thought has a non-empty answer.
        for node_id, data in G.nodes(data=True):
            thought = data['data']
            if thought.answer and len(thought.answer.strip()) > 0:
                print(f"[DEBUG] Termination: Thought {node_id} has produced an answer.")
                return True

        # Estimate token usage based on the content length of all thoughts.
        # Here we assume an approximate conversion (e.g. 4 characters per token).
        total_chars = sum(len(data['data'].content) for _, data in G.nodes(data=True))
        token_usage = total_chars / 4  # approximate token count

        print(f"[DEBUG] Estimated token usage: {token_usage} tokens (Budget: {self.token_budget} tokens).")
        if token_usage >= self.token_budget:
            print(f"[DEBUG] Termination: Token budget reached or exceeded.")
            return True

        return False

    def forward(self, query):
        """Run the AGoT algorithm on the query."""
        final_answer, graph = self._recursive_agot(query, [])
        return {
            "answer": final_answer,
            "graph": graph
        }

    def visualize_graph(self, graph, figsize=(12, 8)):
        """Visualize the graph of thoughts."""
        plt.figure(figsize=figsize)

        # Create position layout
        pos = {}
        layer_counts = {}

        for node_id, data in graph.nodes(data=True):
            thought = data['data']
            layer = thought.layer

            if layer not in layer_counts:
                layer_counts[layer] = 0

            # Position nodes in layers
            pos[node_id] = (layer, -layer_counts[layer])
            layer_counts[layer] += 1

        # Draw nodes
        node_colors = []
        for node_id, data in graph.nodes(data=True):
            thought = data['data']
            if thought.is_complex:
                node_colors.append('orange')
            else:
                node_colors.append('skyblue')

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=700, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrows=True)

        # Draw labels
        labels = {}
        for node_id, data in graph.nodes(data=True):
            thought = data['data']
            labels[node_id] = f"{node_id}\n{thought.content[:20]}..."

        nx.draw_networkx_labels(graph, pos, labels, font_size=8)

        plt.title("Adaptive Graph of Thoughts")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
