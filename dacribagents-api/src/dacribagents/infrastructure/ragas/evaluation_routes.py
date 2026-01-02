"""
RAGAS Evaluation API Routes.

Provides endpoints for running RAGAS evaluations through the API.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field


router = APIRouter(prefix="/eval", tags=["RAGAS Evaluation"])


# =============================================================================
# Request/Response Models
# =============================================================================


class EvaluateSampleRequest(BaseModel):
    """Request to evaluate a single RAG interaction."""
    question: str = Field(..., description="The user's question")
    answer: str = Field(..., description="The generated answer")
    contexts: list[str] = Field(..., description="Retrieved context documents")
    ground_truth: str | None = Field(default=None, description="Optional ground truth answer")


class EvaluateSampleResponse(BaseModel):
    """Response from single sample evaluation."""
    question: str
    faithfulness: dict[str, Any] | None
    answer_relevancy: dict[str, Any] | None
    context_precision: dict[str, Any] | None
    context_recall: dict[str, Any] | None
    overall_score: float


class EvaluateRAGRequest(BaseModel):
    """Request to evaluate a RAG query end-to-end."""
    query: str = Field(..., description="Query to run through RAG")
    ground_truth: str | None = Field(default=None, description="Optional ground truth")
    collection: str = Field(default="email_chunks_v1")
    filter_expr: str | None = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)


class BatchEvaluationRequest(BaseModel):
    """Request for batch evaluation."""
    questions: list[dict[str, Any]] = Field(
        ...,
        description="List of questions with optional ground_truth",
        example=[
            {"question": "What is the fence height limit?", "ground_truth": "6 feet"},
            {"question": "What is ARC approval?"},
        ]
    )
    collection: str = Field(default="email_chunks_v1")
    filter_expr: str | None = Field(default=None)
    dataset_name: str = Field(default="api_evaluation")


class BatchEvaluationResponse(BaseModel):
    """Response from batch evaluation."""
    dataset_name: str
    total_samples: int
    aggregate_metrics: dict[str, float]
    problem_samples: dict[str, list[int]]
    recommendations: list[str]


class MetricExplainRequest(BaseModel):
    """Request to explain a specific metric."""
    metric: str = Field(..., description="Metric name: faithfulness, answer_relevancy, context_precision, context_recall")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/sample", response_model=EvaluateSampleResponse)
async def evaluate_sample(request: EvaluateSampleRequest) -> EvaluateSampleResponse:
    """
    Evaluate a single RAG interaction.
    
    Provide the question, answer, and contexts to get RAGAS metrics.
    
    Example:
    ```json
    {
        "question": "What is the maximum fence height?",
        "answer": "The maximum fence height is 6 feet according to the CC&Rs.",
        "contexts": [
            "Section 4.2: Fences shall not exceed 6 feet in height...",
            "ARC approval is required for all fence installations..."
        ],
        "ground_truth": "6 feet"
    }
    ```
    """
    from .ragas_metrics import RAGASMetrics, EvaluationSample
    
    logger.info(f"Evaluating sample: {request.question[:50]}...")
    
    try:
        metrics = RAGASMetrics()
        
        results = {}
        overall_scores = []
        
        # Faithfulness
        try:
            faith = await metrics.evaluate_faithfulness(
                question=request.question,
                answer=request.answer,
                contexts=request.contexts,
            )
            results["faithfulness"] = {"score": faith.score, "details": faith.details}
            overall_scores.append(faith.score)
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            results["faithfulness"] = None
        
        # Answer Relevancy
        try:
            rel = await metrics.evaluate_answer_relevancy(
                question=request.question,
                answer=request.answer,
            )
            results["answer_relevancy"] = {"score": rel.score, "details": rel.details}
            overall_scores.append(rel.score)
        except Exception as e:
            logger.error(f"Answer relevancy evaluation failed: {e}")
            results["answer_relevancy"] = None
        
        # Context Precision
        try:
            prec = await metrics.evaluate_context_precision(
                question=request.question,
                contexts=request.contexts,
            )
            results["context_precision"] = {"score": prec.score, "details": prec.details}
            overall_scores.append(prec.score)
        except Exception as e:
            logger.error(f"Context precision evaluation failed: {e}")
            results["context_precision"] = None
        
        # Context Recall (only if ground_truth provided)
        if request.ground_truth:
            try:
                recall = await metrics.evaluate_context_recall(
                    question=request.question,
                    contexts=request.contexts,
                    ground_truth=request.ground_truth,
                )
                results["context_recall"] = {"score": recall.score, "details": recall.details}
                overall_scores.append(recall.score)
            except Exception as e:
                logger.error(f"Context recall evaluation failed: {e}")
                results["context_recall"] = None
        else:
            results["context_recall"] = None
        
        overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        return EvaluateSampleResponse(
            question=request.question,
            faithfulness=results.get("faithfulness"),
            answer_relevancy=results.get("answer_relevancy"),
            context_precision=results.get("context_precision"),
            context_recall=results.get("context_recall"),
            overall_score=overall,
        )
        
    except Exception as e:
        logger.exception(f"Sample evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag")
async def evaluate_rag_query(request: EvaluateRAGRequest) -> dict[str, Any]:
    """
    Run a query through RAG and evaluate the result.
    
    This endpoint:
    1. Runs your query through the RAG pipeline
    2. Evaluates the result with RAGAS metrics
    3. Returns both the RAG response and evaluation metrics
    
    Example:
    ```json
    {
        "query": "What is the maximum fence height?",
        "filter_expr": "account_id == \"workmail-hoa\"",
        "top_k": 5
    }
    ```
    """
    from .ragas_metrics import RAGASMetrics
    
    logger.info(f"Evaluating RAG query: {request.query[:50]}...")
    
    try:
        # Run RAG query
        from dacribagents.infrastructure.agentic_rag import get_agentic_rag
        
        rag = get_agentic_rag(
            collection_name=request.collection,
            filter_expr=request.filter_expr,
        )
        
        result = await rag.query(request.query, top_k=request.top_k)
        
        # Extract contexts
        contexts = [chunk.content for chunk in result.retrieved_chunks]
        
        # Run RAGAS evaluation
        metrics = RAGASMetrics()
        
        evaluation = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }
        
        try:
            faith = await metrics.evaluate_faithfulness(
                request.query, result.response, contexts
            )
            evaluation["faithfulness"] = {"score": faith.score, "passed": faith.passed}
        except Exception as e:
            logger.error(f"Faithfulness failed: {e}")
        
        try:
            rel = await metrics.evaluate_answer_relevancy(
                request.query, result.response
            )
            evaluation["answer_relevancy"] = {"score": rel.score, "passed": rel.passed}
        except Exception as e:
            logger.error(f"Answer relevancy failed: {e}")
        
        try:
            prec = await metrics.evaluate_context_precision(
                request.query, contexts
            )
            evaluation["context_precision"] = {"score": prec.score, "passed": prec.passed}
        except Exception as e:
            logger.error(f"Context precision failed: {e}")
        
        if request.ground_truth:
            try:
                recall = await metrics.evaluate_context_recall(
                    request.query, contexts, request.ground_truth
                )
                evaluation["context_recall"] = {"score": recall.score, "passed": recall.passed}
            except Exception as e:
                logger.error(f"Context recall failed: {e}")
        
        # Calculate overall
        scores = [v["score"] for v in evaluation.values() if v is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "query": request.query,
            "rag_result": {
                "response": result.response,
                "is_grounded": result.is_grounded,
                "confidence": result.confidence,
                "iterations": result.iterations,
                "sources": [
                    {"source": c.source, "score": c.score}
                    for c in result.retrieved_chunks[:5]
                ],
            },
            "evaluation": evaluation,
            "overall_score": overall_score,
            "passed": overall_score >= 0.7,
        }
        
    except Exception as e:
        logger.exception(f"RAG evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchEvaluationResponse)
async def batch_evaluation(request: BatchEvaluationRequest) -> BatchEvaluationResponse:
    """
    Run batch evaluation on multiple questions.
    
    Example:
    ```json
    {
        "questions": [
            {"question": "What is the fence height limit?"},
            {"question": "What is ARC approval?"},
            {"question": "Can I have chickens?"}
        ],
        "filter_expr": "account_id == \"workmail-hoa\"",
        "dataset_name": "hoa_evaluation"
    }
    ```
    """
    from .evaluation_pipeline import RAGASEvaluationPipeline
    
    logger.info(f"Starting batch evaluation: {request.dataset_name} ({len(request.questions)} questions)")
    
    try:
        pipeline = RAGASEvaluationPipeline(
            collection_name=request.collection,
            filter_expr=request.filter_expr,
            output_dir="/tmp/eval_results",
        )
        
        report = await pipeline.run_full_evaluation(
            questions=request.questions,
            dataset_name=request.dataset_name,
        )
        
        # Generate recommendations
        recommendations = []
        if report.avg_faithfulness < 0.7:
            recommendations.append("Improve grounding - answers contain unsupported claims")
        if report.avg_context_precision < 0.7:
            recommendations.append("Improve retrieval - too many irrelevant documents")
        if report.avg_answer_relevancy < 0.7:
            recommendations.append("Improve answer generation - responses not addressing questions")
        if report.avg_context_recall < 0.7:
            recommendations.append("Improve recall - missing relevant documents")
        
        if not recommendations:
            recommendations.append("All metrics passing - system performing well")
        
        return BatchEvaluationResponse(
            dataset_name=request.dataset_name,
            total_samples=len(request.questions),
            aggregate_metrics={
                "faithfulness": report.avg_faithfulness,
                "answer_relevancy": report.avg_answer_relevancy,
                "context_precision": report.avg_context_precision,
                "context_recall": report.avg_context_recall,
                "overall": report.overall_score,
            },
            problem_samples={
                "low_faithfulness": report.low_faithfulness_samples,
                "low_relevancy": report.low_relevancy_samples,
                "low_precision": report.low_precision_samples,
                "low_recall": report.low_recall_samples,
            },
            recommendations=recommendations,
        )
        
    except Exception as e:
        logger.exception(f"Batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def list_metrics() -> dict[str, Any]:
    """
    List available RAGAS metrics and their descriptions.
    
    Useful for understanding what each metric measures.
    """
    return {
        "metrics": [
            {
                "name": "faithfulness",
                "description": "Measures if the answer is supported by the retrieved context",
                "range": "0.0 - 1.0",
                "interpretation": "Higher is better. Low scores indicate hallucination.",
                "requires_ground_truth": False,
                "formula": "supported_claims / total_claims",
            },
            {
                "name": "answer_relevancy",
                "description": "Measures if the answer addresses the question",
                "range": "0.0 - 1.0",
                "interpretation": "Higher is better. Low scores indicate off-topic answers.",
                "requires_ground_truth": False,
                "formula": "semantic_similarity(question, generated_questions_from_answer)",
            },
            {
                "name": "context_precision",
                "description": "Measures if retrieved documents are relevant",
                "range": "0.0 - 1.0",
                "interpretation": "Higher is better. Low scores indicate noisy retrieval.",
                "requires_ground_truth": False,
                "formula": "weighted_relevant_docs / total_docs (higher rank = higher weight)",
            },
            {
                "name": "context_recall",
                "description": "Measures if all necessary information was retrieved",
                "range": "0.0 - 1.0",
                "interpretation": "Higher is better. Low scores indicate missing documents.",
                "requires_ground_truth": True,
                "formula": "facts_found_in_context / facts_in_ground_truth",
            },
        ],
        "passing_threshold": 0.7,
        "note": "Context Recall requires ground truth answers to evaluate.",
    }


@router.post("/explain")
async def explain_metric(request: MetricExplainRequest) -> dict[str, Any]:
    """
    Get detailed explanation of a specific metric.
    """
    explanations = {
        "faithfulness": {
            "name": "Faithfulness",
            "what_it_measures": "Whether the answer only contains information that can be verified from the retrieved context.",
            "why_it_matters": "Low faithfulness indicates the model is hallucinating - making up information not in the source documents.",
            "how_to_improve": [
                "Add explicit grounding instructions in prompts",
                "Increase relevance threshold for context filtering",
                "Add citation requirements",
                "Use smaller, more focused context chunks",
            ],
            "interpretation": {
                "0.9-1.0": "Excellent - Answer is well grounded",
                "0.7-0.9": "Good - Minor unsupported claims",
                "0.5-0.7": "Fair - Some hallucination present",
                "0.0-0.5": "Poor - Significant hallucination",
            },
        },
        "answer_relevancy": {
            "name": "Answer Relevancy",
            "what_it_measures": "Whether the answer actually addresses what the user asked.",
            "why_it_matters": "Low relevancy means users aren't getting answers to their questions, even if the information is factually correct.",
            "how_to_improve": [
                "Improve query understanding/reformulation",
                "Add explicit instructions to address the question directly",
                "Use query analysis to identify question type",
                "Consider multi-step reasoning for complex questions",
            ],
            "interpretation": {
                "0.9-1.0": "Excellent - Answer directly addresses question",
                "0.7-0.9": "Good - Answer mostly relevant",
                "0.5-0.7": "Fair - Answer partially addresses question",
                "0.0-0.5": "Poor - Answer is off-topic",
            },
        },
        "context_precision": {
            "name": "Context Precision",
            "what_it_measures": "Whether the retrieved documents are actually relevant to the question.",
            "why_it_matters": "Low precision means the retrieval is noisy - returning irrelevant documents that may confuse the generator.",
            "how_to_improve": [
                "Improve query reformulation",
                "Increase relevance score thresholds",
                "Use metadata filters to narrow search",
                "Review embedding quality",
                "Improve document chunking strategy",
            ],
            "interpretation": {
                "0.9-1.0": "Excellent - All retrieved docs are relevant",
                "0.7-0.9": "Good - Most docs are relevant",
                "0.5-0.7": "Fair - Mixed relevance",
                "0.0-0.5": "Poor - Mostly irrelevant docs",
            },
        },
        "context_recall": {
            "name": "Context Recall",
            "what_it_measures": "Whether all the information needed to answer the question was retrieved.",
            "why_it_matters": "Low recall means important documents are being missed, leading to incomplete or incorrect answers.",
            "how_to_improve": [
                "Increase top_k for retrieval",
                "Use query expansion",
                "Implement hybrid search (semantic + keyword)",
                "Review document chunking boundaries",
                "Ensure all relevant documents are indexed",
            ],
            "interpretation": {
                "0.9-1.0": "Excellent - All needed info retrieved",
                "0.7-0.9": "Good - Most needed info retrieved",
                "0.5-0.7": "Fair - Some info missing",
                "0.0-0.5": "Poor - Critical info missing",
            },
            "note": "Requires ground truth answer to evaluate",
        },
    }
    
    metric = request.metric.lower()
    if metric not in explanations:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric: {metric}. Available: {list(explanations.keys())}"
        )
    
    return explanations[metric]


@router.get("/test-questions")
async def get_test_questions() -> dict[str, Any]:
    """
    Get pre-defined test questions for evaluation.
    
    These can be used as a starting point for your evaluation dataset.
    """
    from .evaluation_pipeline import (
        HOA_TEST_QUESTIONS,
        MULTI_HOP_QUESTIONS,
        EDGE_CASE_QUESTIONS,
    )
    
    return {
        "hoa_questions": HOA_TEST_QUESTIONS,
        "multi_hop_questions": MULTI_HOP_QUESTIONS,
        "edge_case_questions": EDGE_CASE_QUESTIONS,
        "total": len(HOA_TEST_QUESTIONS) + len(MULTI_HOP_QUESTIONS) + len(EDGE_CASE_QUESTIONS),
        "usage": "Use these questions with POST /eval/batch to run evaluation",
    }