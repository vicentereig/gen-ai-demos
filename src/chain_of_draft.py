import os

import dspy
from typing import List, Dict, Any
from dotenv import load_dotenv



class ChainOfDraft(dspy.Module):
    """
    Chain of Draft (CoD) implementation using DSPy framework.

    CoD encourages models to generate minimalistic yet informative intermediate
    reasoning steps, reducing verbosity and token usage while maintaining or
    improving reasoning quality.
    """

    def __init__(self, max_words_per_step: int = 5):
        """
        Initialize the Chain of Draft module.

        Args:
            max_words_per_step: Maximum words per reasoning step (default: 5)
        """
        super().__init__()
        self.max_words_per_step = max_words_per_step

        # Define the signature for our Chain of Draft reasoner
        class CoD(dspy.Signature):
            """Signature for Chain of Draft reasoning."""
            question = dspy.InputField(desc="The question to be answered")
            draft_steps = dspy.OutputField(
                desc=f"Concise reasoning steps, with maximum {max_words_per_step} words per step")
            answer = dspy.OutputField(desc="The final answer to the question")

        # Create the prompt for Chain of Draft
        self.cod_prompt = f"""
        Think step by step to solve the given problem, but only keep a minimum draft for each thinking step, 
        with {self.max_words_per_step} words at most for each step.
        """

        # Create the Chain of Thought with our prompt
        self.cod_reasoning = dspy.ChainOfThought(CoD, prompt=self.cod_prompt)

    def forward(self, question: str) -> Dict[str, Any]:
        """
        Run the Chain of Draft reasoning process on a given question.

        Args:
            question: The question to reason about and answer

        Returns:
            Dict containing the question, draft reasoning steps, and final answer
        """
        result = self.cod_reasoning(question=question)
        return {
            "question": question,
            "draft_steps": result.draft_steps,
            "answer": result.answer
        }


class ChainOfDraftWithExamples(ChainOfDraft):
    """
    Chain of Draft with few-shot examples to help guide the model's reasoning format.
    """

    def __init__(self, max_words_per_step: int = 5, examples: List[Dict] = None):
        """
        Initialize with optional few-shot examples.

        Args:
            max_words_per_step: Maximum words per reasoning step
            examples: List of example dictionaries with 'question', 'draft_steps', and 'answer' keys
        """
        super().__init__(max_words_per_step)

        if examples:
            dspy_examples = []
            for ex in examples:
                dspy_examples.append(
                    dspy.Example(
                        question=ex["question"],
                        draft_steps=ex["draft_steps"],
                        answer=ex["answer"]
                    )
                )
            self.cod_reasoning.demos = dspy_examples


# Example usage
def demonstrate_chain_of_draft():
    """Demonstrate the Chain of Draft framework with examples."""

    # Define a few-shot example for arithmetic
    arithmetic_examples = [
        {
            "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "draft_steps": "20 - x = 12; x = 8",
            "answer": "8 lollipops"
        },
        {
            "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
            "draft_steps": "5 + (2 × 3) = 5 + 6 = 11",
            "answer": "11 tennis balls"
        }
    ]

    # Create Chain of Draft with examples
    cod = ChainOfDraftWithExamples(max_words_per_step=5, examples=arithmetic_examples)

    # Test with a new question
    result = cod.forward(
        "Sarah has 27 apples. She gives 1/3 of her apples to John. How many apples does Sarah have left?")

    # Print the result
    print("Question:", result["question"])
    print("Draft Steps:", result["draft_steps"])
    print("Answer:", result["answer"])
    print()

    # Calculate token savings compared to CoT (estimated)
    draft_tokens = len(result["draft_steps"].split())
    estimated_cot_tokens = draft_tokens * 5  # Rough estimate based on paper's findings

    print(f"Estimated tokens used in draft: {draft_tokens}")
    print(f"Estimated tokens in equivalent CoT: {estimated_cot_tokens}")
    print(f"Estimated token reduction: {(1 - draft_tokens / estimated_cot_tokens) * 100:.1f}%")


def demonstrate_chain_of_draft_v2():
    """Demonstrate the Chain of Draft framework with examples."""

    # Define a few-shot example for arithmetic
    arithmetic_examples = [

    ]

    # Create Chain of Draft with examples
    cod = ChainOfDraftWithExamples(max_words_per_step=5, examples=arithmetic_examples)

    # Test with a new question
    result = cod.forward(
        "What's the best opening move in Catan? Explain in very high detail 3 potential scenarios where it showcases the best performance. I am a beginner to catan and I want instructions on how to start super strong.")

    # Print the result
    print("Question:", result["question"])
    print("Draft Steps:", result["draft_steps"])
    print("Answer:", result["answer"])
    print()

    # Calculate token savings compared to CoT (estimated)
    draft_tokens = len(result["draft_steps"].split())
    estimated_cot_tokens = draft_tokens * 5  # Rough estimate based on paper's findings

    print(f"Estimated tokens used in draft: {draft_tokens}")
    print(f"Estimated tokens in equivalent CoT: {estimated_cot_tokens}")
    print(f"Estimated token reduction: {(1 - draft_tokens / estimated_cot_tokens) * 100:.1f}%")

if __name__ == "__main__":
    load_dotenv()
    load_dotenv('../.secrets')
    lm = dspy.LM('gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)
    # demonstrate_chain_of_draft()
    demonstrate_chain_of_draft_v2()