"""
RAGAS Evaluation Pipeline.

Integrates RAGAS evaluation with the Agentic RAG system.

This pipeline:
1. Generates evaluation datasets from your RAG system
2. Runs RAGAS evaluation on the datasets
3. Provides actionable insights for improvement

Workflow:
1. Define test questions (with optional ground truth)
2. Run RAG system to get answers and contexts
3. Evaluate with RAGAS metrics
4. Analyze results and identify issues
5. Iterate on improvements
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .ragas_metrics import (
    EvaluationDataset,
    EvaluationReport,
    EvaluationSample,
    RAGASEvaluator,
    create_evaluation_dataset,
    create_evaluation_sample,
)


# =============================================================================
# Evaluation Dataset Generator
# =============================================================================


class EvaluationDatasetGenerator:
    """
    Generates evaluation datasets by running queries through the RAG system.
    
    This automates the process of:
    1. Running queries through RAG
    2. Capturing contexts and answers
    3. Creating evaluation samples
    """
    
    def __init__(
        self,
        collection_name: str = "email_chunks_v1",
        filter_expr: str | None = None,
    ):
        self.collection_name = collection_name
        self.filter_expr = filter_expr
        self._rag = None
    
    @property
    def rag(self):
        """Lazy load RAG pipeline."""
        if self._rag is None:
            from dacribagents.infrastructure.agentic_rag import get_agentic_rag
            self._rag = get_agentic_rag(
                collection_name=self.collection_name,
                filter_expr=self.filter_expr,
            )
        return self._rag
    
    async def generate_sample(
        self,
        question: str,
        ground_truth: str | None = None,
        top_k: int = 5,
    ) -> EvaluationSample:
        """
        Generate a single evaluation sample by running the RAG pipeline.
        
        Args:
            question: The question to evaluate
            ground_truth: Optional human-verified answer
            top_k: Number of documents to retrieve
        
        Returns:
            EvaluationSample with question, contexts, answer
        """
        # Run RAG query
        result = await self.rag.query(question, top_k=top_k)
        
        # Extract contexts from retrieved chunks
        contexts = [chunk.content for chunk in result.retrieved_chunks]
        
        return create_evaluation_sample(
            question=question,
            contexts=contexts,
            answer=result.response,
            ground_truth=ground_truth,
            # Metadata for analysis
            is_grounded=result.is_grounded,
            confidence=result.confidence,
            iterations=result.iterations,
            strategy=result.strategy_used.value,
            reformulated_query=result.reformulated_query,
        )
    
    async def generate_dataset(
        self,
        questions: list[dict[str, Any]],
        name: str = "evaluation_dataset",
        description: str = "",
    ) -> EvaluationDataset:
        """
        Generate a complete evaluation dataset.
        
        Args:
            questions: List of dicts with 'question' and optional 'ground_truth'
            name: Dataset name
            description: Dataset description
        
        Returns:
            EvaluationDataset ready for RAGAS evaluation
        """
        samples = []
        
        for i, q in enumerate(questions):
            logger.info(f"Generating sample {i+1}/{len(questions)}: {q['question'][:50]}...")
            
            sample = await self.generate_sample(
                question=q["question"],
                ground_truth=q.get("ground_truth"),
                top_k=q.get("top_k", 5),
            )
            samples.append(sample)
        
        return create_evaluation_dataset(
            name=name,
            description=description,
            samples=samples,
        )


# =============================================================================
# Pre-defined Test Questions
# =============================================================================


# HOA Domain Test Questions
HOA_TEST_QUESTIONS = [
    {
        "question": "What is the maximum fence height allowed?",
        "ground_truth": None,  # Will be filled based on your actual documents
    },
    {
        "question": "What is the ARC approval process?",
        "ground_truth": None,
    },
    {
        "question": "What exterior paint colors are approved?",
        "ground_truth": None,
    },
    {
        "question": "What modifications require ARC approval?",
        "ground_truth": None,
    },
    {
        "question": "How long does ARC approval take?",
        "ground_truth": None,
    },
    {
        "question": "What are the landscaping requirements?",
        "ground_truth": None,
    },
    {
        "question": "What happens if I violate the CC&Rs?",
        "ground_truth": None,
    },
    {
        "question": "Can I have a shed in my backyard?",
        "ground_truth": None,
    },
    {
        "question": "What are the rules for holiday decorations?",
        "ground_truth": None,
    },
    {
        "question": "Can I park an RV in my driveway?",
        "ground_truth": None,
    },
]

# Multi-hop Questions (require combining multiple pieces of information)
MULTI_HOP_QUESTIONS = [
    {
        "question": "What modifications need ARC approval and how long does the approval process take?",
        "ground_truth": None,
    },
    {
        "question": "If I want to build a fence, what are the height limits and what materials are approved?",
        "ground_truth": None,
    },
    {
        "question": "What are the paint color requirements and how do I get approval to repaint my house?",
        "ground_truth": None,
    },
]

# Edge Case Questions (test system robustness)
EDGE_CASE_QUESTIONS = [
    {
        "question": "chickens?",  # Very short query
        "ground_truth": None,
    },
    {
        "question": "What is the exact fine amount for a first-time fence violation?",  # May not have answer
        "ground_truth": None,
    },
    {
        "question": "Can I install solar panels on my roof?",  # Cross-domain (solar + HOA)
        "ground_truth": None,
    },
    {
        "question": "What did the board decide in the last meeting?",  # Time-sensitive
        "ground_truth": None,
    },
]


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================


class RAGASEvaluationPipeline:
    """
    Complete evaluation pipeline for RAG systems.
    
    Workflow:
    1. Generate evaluation datasets
    2. Run RAGAS evaluation
    3. Analyze results
    4. Generate improvement recommendations
    """
    
    def __init__(
        self,
        collection_name: str = "email_chunks_v1",
        filter_expr: str | None = None,
        output_dir: str | Path = "evaluation_results",
    ):
        self.collection_name = collection_name
        self.filter_expr = filter_expr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = EvaluationDatasetGenerator(
            collection_name=collection_name,
            filter_expr=filter_expr,
        )
        self.evaluator = RAGASEvaluator()
    
    async def run_full_evaluation(
        self,
        questions: list[dict[str, Any]] | None = None,
        dataset_name: str = "full_evaluation",
    ) -> EvaluationReport:
        """
        Run complete evaluation pipeline.
        
        Args:
            questions: Questions to evaluate (uses defaults if None)
            dataset_name: Name for the evaluation dataset
        
        Returns:
            Complete EvaluationReport
        """
        # Use default questions if none provided
        if questions is None:
            questions = HOA_TEST_QUESTIONS + MULTI_HOP_QUESTIONS + EDGE_CASE_QUESTIONS
        
        logger.info(f"Starting evaluation with {len(questions)} questions...")
        
        # Step 1: Generate dataset
        logger.info("Step 1: Generating evaluation dataset...")
        dataset = await self.generator.generate_dataset(
            questions=questions,
            name=dataset_name,
            description=f"Evaluation of {self.collection_name} with filter: {self.filter_expr}",
        )
        
        # Save dataset
        dataset_path = self.output_dir / f"{dataset_name}_dataset.json"
        self._save_dataset(dataset, dataset_path)
        
        # Step 2: Run RAGAS evaluation
        logger.info("Step 2: Running RAGAS evaluation...")
        report = await self.evaluator.evaluate_dataset(dataset)
        
        # Step 3: Save and print report
        report_path = self.output_dir / f"{dataset_name}_report.json"
        self.evaluator.export_report(report, report_path)
        self.evaluator.print_report(report)
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(report)
        rec_path = self.output_dir / f"{dataset_name}_recommendations.md"
        self._save_recommendations(recommendations, rec_path)
        
        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
        
        return report
    
    async def run_quick_evaluation(
        self,
        questions: list[str],
        dataset_name: str = "quick_eval",
    ) -> EvaluationReport:
        """
        Run quick evaluation with just question strings (no ground truth).
        """
        question_dicts = [{"question": q} for q in questions]
        return await self.run_full_evaluation(question_dicts, dataset_name)
    
    async def compare_configurations(
        self,
        questions: list[dict[str, Any]],
        configurations: list[dict[str, Any]],
    ) -> dict[str, EvaluationReport]:
        """
        Compare different RAG configurations.
        
        Args:
            questions: Questions to evaluate
            configurations: List of config dicts with 'name', 'filter_expr', etc.
        
        Returns:
            Dict mapping config name to evaluation report
        """
        results = {}
        
        for config in configurations:
            config_name = config.get("name", "unnamed")
            logger.info(f"Evaluating configuration: {config_name}")
            
            # Create generator with this config
            self.generator = EvaluationDatasetGenerator(
                collection_name=config.get("collection_name", self.collection_name),
                filter_expr=config.get("filter_expr"),
            )
            
            report = await self.run_full_evaluation(
                questions=questions,
                dataset_name=f"compare_{config_name}",
            )
            results[config_name] = report
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _save_dataset(self, dataset: EvaluationDataset, path: Path):
        """Save dataset to JSON."""
        data = {
            "name": dataset.name,
            "description": dataset.description,
            "created_at": dataset.created_at,
            "total_samples": dataset.total_samples,
            "samples_with_ground_truth": dataset.samples_with_ground_truth,
            "samples": [
                {
                    "question": s.question,
                    "contexts": s.contexts,
                    "answer": s.answer,
                    "ground_truth": s.ground_truth,
                    "metadata": s.metadata,
                }
                for s in dataset.samples
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _generate_recommendations(self, report: EvaluationReport) -> str:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []
        recommendations.append("# RAG System Improvement Recommendations\n")
        recommendations.append(f"Based on evaluation: {report.dataset_name}")
        recommendations.append(f"Timestamp: {report.timestamp}\n")
        
        recommendations.append("## Summary\n")
        recommendations.append(f"- Overall Score: {report.overall_score:.2f}")
        recommendations.append(f"- Faithfulness: {report.avg_faithfulness:.2f}")
        recommendations.append(f"- Answer Relevancy: {report.avg_answer_relevancy:.2f}")
        recommendations.append(f"- Context Precision: {report.avg_context_precision:.2f}")
        recommendations.append(f"- Context Recall: {report.avg_context_recall:.2f}\n")
        
        recommendations.append("## Priority Improvements\n")
        
        # Faithfulness issues
        if report.avg_faithfulness < 0.7:
            recommendations.append("### 🔴 Critical: Low Faithfulness")
            recommendations.append("The system is generating answers not supported by retrieved documents.\n")
            recommendations.append("**Actions:**")
            recommendations.append("1. Review and strengthen grounding prompts")
            recommendations.append("2. Increase relevance threshold for document filtering")
            recommendations.append("3. Add explicit instructions to only use retrieved information")
            recommendations.append("4. Consider adding citation requirements\n")
        
        # Context Precision issues
        if report.avg_context_precision < 0.7:
            recommendations.append("### 🟠 High Priority: Low Context Precision")
            recommendations.append("The retrieval is returning irrelevant documents.\n")
            recommendations.append("**Actions:**")
            recommendations.append("1. Improve query reformulation")
            recommendations.append("2. Increase the relevance score threshold")
            recommendations.append("3. Consider re-chunking documents with better boundaries")
            recommendations.append("4. Review embedding model quality")
            recommendations.append("5. Add metadata filters to narrow search scope\n")
        
        # Answer Relevancy issues
        if report.avg_answer_relevancy < 0.7:
            recommendations.append("### 🟠 High Priority: Low Answer Relevancy")
            recommendations.append("Answers are not addressing the questions effectively.\n")
            recommendations.append("**Actions:**")
            recommendations.append("1. Improve answer generation prompts")
            recommendations.append("2. Add query analysis to understand user intent")
            recommendations.append("3. Consider multi-step reasoning for complex queries")
            recommendations.append("4. Review if queries need reformulation\n")
        
        # Context Recall issues
        if report.avg_context_recall < 0.7:
            recommendations.append("### 🟡 Medium Priority: Low Context Recall")
            recommendations.append("The system is missing relevant documents.\n")
            recommendations.append("**Actions:**")
            recommendations.append("1. Increase top_k for retrieval")
            recommendations.append("2. Use query expansion techniques")
            recommendations.append("3. Consider hybrid search (semantic + keyword)")
            recommendations.append("4. Review document chunking strategy")
            recommendations.append("5. Ensure all relevant documents are indexed\n")
        
        # All metrics good
        if all([
            report.avg_faithfulness >= 0.7,
            report.avg_answer_relevancy >= 0.7,
            report.avg_context_precision >= 0.7,
            report.avg_context_recall >= 0.7,
        ]):
            recommendations.append("### ✅ All Metrics Passing")
            recommendations.append("The RAG system is performing well.\n")
            recommendations.append("**Next Steps:**")
            recommendations.append("1. Continue monitoring in production")
            recommendations.append("2. Expand test dataset for edge cases")
            recommendations.append("3. Consider A/B testing optimizations")
            recommendations.append("4. Add more domain-specific test questions\n")
        
        # Problem samples
        recommendations.append("## Samples Requiring Investigation\n")
        
        if report.low_faithfulness_samples:
            recommendations.append(f"### Low Faithfulness Samples: {report.low_faithfulness_samples}")
            recommendations.append("Review these samples for hallucination issues.\n")
        
        if report.low_precision_samples:
            recommendations.append(f"### Low Precision Samples: {report.low_precision_samples}")
            recommendations.append("Review retrieval quality for these queries.\n")
        
        if report.low_relevancy_samples:
            recommendations.append(f"### Low Relevancy Samples: {report.low_relevancy_samples}")
            recommendations.append("Review answer generation for these queries.\n")
        
        if report.low_recall_samples:
            recommendations.append(f"### Low Recall Samples: {report.low_recall_samples}")
            recommendations.append("Ensure all relevant documents are being retrieved.\n")
        
        return "\n".join(recommendations)
    
    def _save_recommendations(self, recommendations: str, path: Path):
        """Save recommendations to markdown file."""
        with open(path, "w") as f:
            f.write(recommendations)
        logger.info(f"Recommendations saved to {path}")
    
    def _print_comparison(self, results: dict[str, EvaluationReport]):
        """Print comparison of multiple configurations."""
        print("\n" + "=" * 70)
        print("CONFIGURATION COMPARISON")
        print("=" * 70)
        print(f"\n{'Config':<20} {'Faith.':<10} {'Relev.':<10} {'Prec.':<10} {'Recall':<10} {'Overall':<10}")
        print("-" * 70)
        
        for name, report in results.items():
            print(f"{name:<20} {report.avg_faithfulness:<10.3f} {report.avg_answer_relevancy:<10.3f} "
                  f"{report.avg_context_precision:<10.3f} {report.avg_context_recall:<10.3f} "
                  f"{report.overall_score:<10.3f}")
        
        print("=" * 70)


# =============================================================================
# CLI Runner
# =============================================================================


async def run_evaluation_cli():
    """CLI entry point for running evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on RAG system")
    parser.add_argument("--collection", default="email_chunks_v1", help="Milvus collection name")
    parser.add_argument("--filter", default='account_id == "workmail-hoa"', help="Filter expression")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation with fewer questions")
    
    args = parser.parse_args()
    
    pipeline = RAGASEvaluationPipeline(
        collection_name=args.collection,
        filter_expr=args.filter,
        output_dir=args.output_dir,
    )
    
    if args.quick:
        questions = HOA_TEST_QUESTIONS[:5]
    else:
        questions = HOA_TEST_QUESTIONS + MULTI_HOP_QUESTIONS + EDGE_CASE_QUESTIONS
    
    await pipeline.run_full_evaluation(questions=questions)


if __name__ == "__main__":
    asyncio.run(run_evaluation_cli())