"""
RAGAS Evaluation Module.

Provides comprehensive evaluation for RAG systems using RAGAS metrics:
- Faithfulness: Is the answer grounded in context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Did we get all needed docs?

Usage:
    from dacribagents.infrastructure.ragas import (
        RAGASEvaluator,
        RAGASEvaluationPipeline,
        create_evaluation_sample,
        create_evaluation_dataset,
    )
    
    # Quick evaluation
    evaluator = RAGASEvaluator()
    sample = create_evaluation_sample(
        question="What is the fence height?",
        contexts=["Fences shall not exceed 6 feet..."],
        answer="The maximum fence height is 6 feet.",
    )
    result = await evaluator.evaluate_sample(sample)
    
    # Full pipeline
    pipeline = RAGASEvaluationPipeline(
        collection_name="email_chunks_v1",
        filter_expr='account_id == "workmail-hoa"',
    )
    report = await pipeline.run_full_evaluation()
"""

from .ragas_metrics import (
    EvaluationDataset,
    EvaluationReport,
    EvaluationSample,
    MetricResult,
    RAGASEvaluator,
    RAGASMetrics,
    SampleEvaluation,
    create_evaluation_dataset,
    create_evaluation_sample,
    get_ragas_evaluator,
)

from .evaluation_pipeline import (
    EvaluationDatasetGenerator,
    RAGASEvaluationPipeline,
    HOA_TEST_QUESTIONS,
    MULTI_HOP_QUESTIONS,
    EDGE_CASE_QUESTIONS,
)

from .evaluation_routes import router as evaluation_router

__all__ = [
    # Data Models
    "EvaluationSample",
    "EvaluationDataset",
    "MetricResult",
    "SampleEvaluation",
    "EvaluationReport",
    # Metrics
    "RAGASMetrics",
    # Evaluator
    "RAGASEvaluator",
    "get_ragas_evaluator",
    # Pipeline
    "EvaluationDatasetGenerator",
    "RAGASEvaluationPipeline",
    # Factory Functions
    "create_evaluation_sample",
    "create_evaluation_dataset",
    # Test Questions
    "HOA_TEST_QUESTIONS",
    "MULTI_HOP_QUESTIONS",
    "EDGE_CASE_QUESTIONS",
    # Routes
    "evaluation_router",
]