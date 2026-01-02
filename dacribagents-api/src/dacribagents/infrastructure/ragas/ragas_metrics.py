"""
RAGAS Evaluation Framework for Agentic RAG.

RAGAS (Retrieval Augmented Generation Assessment) provides metrics to evaluate
RAG systems without requiring ground truth answers for all questions.

Key Metrics:
- Faithfulness: Is the answer supported by the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved documents relevant to the question?
- Context Recall: Did we retrieve all necessary information?

This framework is essential for:
1. Identifying weaknesses in your RAG pipeline
2. Comparing different retrieval strategies
3. Measuring impact of changes (A/B testing)
4. Continuous monitoring in production

NVIDIA Exam Focus:
- Understanding each metric's purpose
- When to use which metric
- How metrics guide system improvements
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


# =============================================================================
# Data Models
# =============================================================================


class EvaluationSample(BaseModel):
    """
    A single evaluation sample.
    
    For RAGAS evaluation, we need:
    - question: The user's query
    - contexts: Retrieved documents (list of strings)
    - answer: The generated response
    - ground_truth: (Optional) Human-verified correct answer
    
    Ground truth is optional because RAGAS can evaluate without it,
    but having it enables Context Recall measurement.
    """
    question: str
    contexts: list[str]
    answer: str
    ground_truth: str | None = None
    
    # Metadata for analysis
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    """Collection of evaluation samples."""
    name: str
    description: str = ""
    samples: list[EvaluationSample]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Dataset statistics
    total_samples: int = 0
    samples_with_ground_truth: int = 0
    
    def model_post_init(self, __context):
        self.total_samples = len(self.samples)
        self.samples_with_ground_truth = sum(
            1 for s in self.samples if s.ground_truth is not None
        )


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    metric_name: str
    score: float  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Consider passing if score >= 0.7"""
        return self.score >= 0.7


@dataclass
class SampleEvaluation:
    """Complete evaluation of a single sample."""
    sample: EvaluationSample
    faithfulness: MetricResult | None = None
    answer_relevancy: MetricResult | None = None
    context_precision: MetricResult | None = None
    context_recall: MetricResult | None = None
    
    @property
    def overall_score(self) -> float:
        """Average of all available metrics."""
        scores = []
        if self.faithfulness:
            scores.append(self.faithfulness.score)
        if self.answer_relevancy:
            scores.append(self.answer_relevancy.score)
        if self.context_precision:
            scores.append(self.context_precision.score)
        if self.context_recall:
            scores.append(self.context_recall.score)
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report for a dataset."""
    dataset_name: str
    timestamp: str
    sample_evaluations: list[SampleEvaluation]
    
    # Aggregate metrics
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    overall_score: float = 0.0
    
    # Error analysis
    low_faithfulness_samples: list[int] = field(default_factory=list)
    low_relevancy_samples: list[int] = field(default_factory=list)
    low_precision_samples: list[int] = field(default_factory=list)
    low_recall_samples: list[int] = field(default_factory=list)
    
    def compute_aggregates(self):
        """Compute aggregate metrics from sample evaluations."""
        faith_scores = []
        rel_scores = []
        prec_scores = []
        rec_scores = []
        
        for i, eval in enumerate(self.sample_evaluations):
            if eval.faithfulness:
                faith_scores.append(eval.faithfulness.score)
                if eval.faithfulness.score < 0.7:
                    self.low_faithfulness_samples.append(i)
                    
            if eval.answer_relevancy:
                rel_scores.append(eval.answer_relevancy.score)
                if eval.answer_relevancy.score < 0.7:
                    self.low_relevancy_samples.append(i)
                    
            if eval.context_precision:
                prec_scores.append(eval.context_precision.score)
                if eval.context_precision.score < 0.7:
                    self.low_precision_samples.append(i)
                    
            if eval.context_recall:
                rec_scores.append(eval.context_recall.score)
                if eval.context_recall.score < 0.7:
                    self.low_recall_samples.append(i)
        
        self.avg_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
        self.avg_answer_relevancy = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0
        self.avg_context_precision = sum(prec_scores) / len(prec_scores) if prec_scores else 0.0
        self.avg_context_recall = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
        
        all_scores = faith_scores + rel_scores + prec_scores + rec_scores
        self.overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0


# =============================================================================
# RAGAS Metrics Implementation
# =============================================================================


class RAGASMetrics:
    """
    RAGAS metrics implementation using LLM-as-judge.
    
    Each metric uses carefully crafted prompts to evaluate different aspects
    of RAG quality. The LLM acts as an evaluator, providing scores and reasoning.
    
    Key Insight: We use the same LLM for evaluation that we use for generation,
    but with different prompts focused on evaluation rather than answering.
    """
    
    def __init__(self, llm=None):
        """
        Initialize RAGAS metrics.
        
        Args:
            llm: LangChain-compatible LLM. If None, will be lazily loaded.
        """
        self._llm = llm
    
    @property
    def llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from dacribagents.application.agents.general_assistant import create_llm
            from dacribagents.infrastructure.settings import get_settings
            self._llm = create_llm(get_settings())
        return self._llm
    
    # -------------------------------------------------------------------------
    # Faithfulness
    # -------------------------------------------------------------------------
    
    async def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> MetricResult:
        """
        Evaluate Faithfulness: Is the answer supported by the context?
        
        Process:
        1. Extract claims from the answer
        2. For each claim, check if it's supported by the context
        3. Score = (supported claims) / (total claims)
        
        High faithfulness means the answer doesn't hallucinate.
        Low faithfulness indicates the answer contains unsupported claims.
        
        This is the most critical metric for RAG systems.
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Step 1: Extract claims from the answer
        claim_extraction_prompt = ChatPromptTemplate.from_template("""
Given a question and an answer, extract all factual claims made in the answer.
A claim is a statement that can be verified as true or false.

Question: {question}
Answer: {answer}

List each claim on a separate line, numbered. If no claims can be extracted, write "NO_CLAIMS".

Claims:""")
        
        chain = claim_extraction_prompt | self.llm | StrOutputParser()
        claims_text = await chain.ainvoke({"question": question, "answer": answer})
        
        # Parse claims
        claims = []
        for line in claims_text.strip().split("\n"):
            line = line.strip()
            if line and not line.upper() == "NO_CLAIMS":
                # Remove numbering
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                    line = line.split(")", 1)[-1].strip()
                if line:
                    claims.append(line)
        
        if not claims:
            return MetricResult(
                metric_name="faithfulness",
                score=1.0,  # No claims = nothing to be unfaithful about
                details={"claims": [], "reason": "No factual claims in answer"}
            )
        
        # Step 2: Verify each claim against context
        context_text = "\n\n".join(contexts)
        
        verification_prompt = ChatPromptTemplate.from_template("""
Given the following context and a claim, determine if the claim is supported by the context.

Context:
{context}

Claim: {claim}

Is this claim supported by the context? Answer with:
- SUPPORTED: if the context explicitly or implicitly supports the claim
- NOT_SUPPORTED: if the context contradicts or doesn't mention the claim
- PARTIALLY: if the context partially supports the claim

Answer (SUPPORTED/NOT_SUPPORTED/PARTIALLY):""")
        
        chain = verification_prompt | self.llm | StrOutputParser()
        
        supported_count = 0
        claim_results = []
        
        for claim in claims:
            result = await chain.ainvoke({"context": context_text, "claim": claim})
            result = result.strip().upper()
            
            if "SUPPORTED" in result and "NOT" not in result:
                supported_count += 1
                verdict = "supported"
            elif "PARTIAL" in result:
                supported_count += 0.5
                verdict = "partial"
            else:
                verdict = "not_supported"
            
            claim_results.append({"claim": claim, "verdict": verdict})
        
        score = supported_count / len(claims)
        
        return MetricResult(
            metric_name="faithfulness",
            score=score,
            details={
                "claims": claim_results,
                "total_claims": len(claims),
                "supported_claims": supported_count,
            }
        )
    
    # -------------------------------------------------------------------------
    # Answer Relevancy
    # -------------------------------------------------------------------------
    
    async def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> MetricResult:
        """
        Evaluate Answer Relevancy: Does the answer address the question?
        
        Process:
        1. Generate N questions that the answer would be appropriate for
        2. Measure semantic similarity between original question and generated questions
        3. Score = average similarity
        
        High relevancy means the answer directly addresses what was asked.
        Low relevancy indicates the answer may be off-topic or incomplete.
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Generate questions the answer would be appropriate for
        question_gen_prompt = ChatPromptTemplate.from_template("""
Given the following answer, generate 3 different questions that this answer would appropriately address.
The questions should be specific and directly answerable by this answer.

Answer: {answer}

Generate 3 questions, one per line:""")
        
        chain = question_gen_prompt | self.llm | StrOutputParser()
        gen_questions_text = await chain.ainvoke({"answer": answer})
        
        # Parse generated questions
        gen_questions = []
        for line in gen_questions_text.strip().split("\n"):
            line = line.strip()
            if line:
                # Remove numbering
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                    line = line.split(")", 1)[-1].strip()
                if line and "?" in line:
                    gen_questions.append(line)
        
        if not gen_questions:
            return MetricResult(
                metric_name="answer_relevancy",
                score=0.5,  # Can't evaluate, assume moderate
                details={"reason": "Could not generate comparison questions"}
            )
        
        # Compute semantic similarity using embeddings
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Embed original question and generated questions
            original_embedding = model.encode(question)
            gen_embeddings = model.encode(gen_questions)
            
            # Compute cosine similarities
            from numpy import dot
            from numpy.linalg import norm
            
            similarities = []
            for gen_emb in gen_embeddings:
                sim = dot(original_embedding, gen_emb) / (norm(original_embedding) * norm(gen_emb))
                similarities.append(float(sim))
            
            score = sum(similarities) / len(similarities)
            
            return MetricResult(
                metric_name="answer_relevancy",
                score=score,
                details={
                    "generated_questions": gen_questions,
                    "similarities": similarities,
                    "avg_similarity": score,
                }
            )
        except Exception as e:
            logger.warning(f"Embedding-based relevancy failed, falling back to LLM: {e}")
            return await self._evaluate_answer_relevancy_llm(question, answer)
    
    async def _evaluate_answer_relevancy_llm(
        self,
        question: str,
        answer: str,
    ) -> MetricResult:
        """Fallback LLM-based answer relevancy evaluation."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template("""
Evaluate how well the answer addresses the question.

Question: {question}
Answer: {answer}

Score the relevancy from 0 to 10 where:
- 10: Answer directly and completely addresses the question
- 7-9: Answer mostly addresses the question with minor gaps
- 4-6: Answer partially addresses the question
- 1-3: Answer barely relates to the question
- 0: Answer is completely irrelevant

Provide your score and brief reasoning.
Format: SCORE: X/10 | REASON: ...

Evaluation:""")
        
        chain = prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke({"question": question, "answer": answer})
        
        # Parse score
        try:
            score_part = result.split("|")[0]
            score_str = score_part.replace("SCORE:", "").replace("/10", "").strip()
            score = float(score_str) / 10.0
        except (ValueError, IndexError):
            score = 0.5
        
        return MetricResult(
            metric_name="answer_relevancy",
            score=score,
            details={"llm_evaluation": result}
        )
    
    # -------------------------------------------------------------------------
    # Context Precision
    # -------------------------------------------------------------------------
    
    async def evaluate_context_precision(
        self,
        question: str,
        contexts: list[str],
    ) -> MetricResult:
        """
        Evaluate Context Precision: Are the retrieved documents relevant?
        
        Process:
        1. For each retrieved document, check if it's relevant to the question
        2. Score using precision@k: relevant docs weighted by position
        
        High precision means we're not retrieving irrelevant documents.
        Low precision indicates noisy retrieval.
        
        This metric helps identify retrieval quality issues.
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        if not contexts:
            return MetricResult(
                metric_name="context_precision",
                score=0.0,
                details={"reason": "No contexts retrieved"}
            )
        
        relevance_prompt = ChatPromptTemplate.from_template("""
Given a question and a document, determine if the document is relevant to answering the question.

Question: {question}

Document:
{document}

Is this document relevant to answering the question?
Answer YES if the document contains information that could help answer the question.
Answer NO if the document is not useful for answering the question.

Answer (YES/NO):""")
        
        chain = relevance_prompt | self.llm | StrOutputParser()
        
        relevance_scores = []
        
        for i, context in enumerate(contexts):
            result = await chain.ainvoke({
                "question": question,
                "document": context[:2000],  # Truncate for efficiency
            })
            
            is_relevant = "YES" in result.upper()
            relevance_scores.append({
                "position": i + 1,
                "relevant": is_relevant,
                "document_preview": context[:200] + "..." if len(context) > 200 else context
            })
        
        # Calculate precision@k with position weighting
        # Documents ranked higher should be more important
        weighted_sum = 0
        weight_total = 0
        
        for i, score in enumerate(relevance_scores):
            weight = 1.0 / (i + 1)  # Higher weight for higher-ranked docs
            weight_total += weight
            if score["relevant"]:
                weighted_sum += weight
        
        precision_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        # Also calculate simple precision
        simple_precision = sum(1 for s in relevance_scores if s["relevant"]) / len(relevance_scores)
        
        return MetricResult(
            metric_name="context_precision",
            score=precision_score,
            details={
                "relevance_scores": relevance_scores,
                "weighted_precision": precision_score,
                "simple_precision": simple_precision,
                "total_contexts": len(contexts),
                "relevant_contexts": sum(1 for s in relevance_scores if s["relevant"]),
            }
        )
    
    # -------------------------------------------------------------------------
    # Context Recall
    # -------------------------------------------------------------------------
    
    async def evaluate_context_recall(
        self,
        question: str,
        contexts: list[str],
        ground_truth: str,
    ) -> MetricResult:
        """
        Evaluate Context Recall: Did we retrieve all necessary information?
        
        Process:
        1. Extract key facts from the ground truth answer
        2. Check if each fact is present in the retrieved contexts
        3. Score = (facts found in context) / (total facts in ground truth)
        
        High recall means we retrieved all the information needed.
        Low recall indicates we're missing relevant documents.
        
        NOTE: Requires ground truth answer to evaluate.
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Extract key facts from ground truth
        fact_extraction_prompt = ChatPromptTemplate.from_template("""
Given a question and its correct answer, extract the key facts that are essential to answering the question.

Question: {question}
Answer: {ground_truth}

List each key fact on a separate line. Focus on factual information, not opinions or hedging.

Key facts:""")
        
        chain = fact_extraction_prompt | self.llm | StrOutputParser()
        facts_text = await chain.ainvoke({
            "question": question,
            "ground_truth": ground_truth
        })
        
        # Parse facts
        facts = []
        for line in facts_text.strip().split("\n"):
            line = line.strip()
            if line and not line.lower().startswith("key fact"):
                # Remove numbering/bullets
                if line[0] in "0123456789-•*":
                    line = line[1:].strip()
                    if line and line[0] in ".):":
                        line = line[1:].strip()
                if line:
                    facts.append(line)
        
        if not facts:
            return MetricResult(
                metric_name="context_recall",
                score=1.0,  # No facts to recall
                details={"reason": "No key facts extracted from ground truth"}
            )
        
        # Check if each fact is present in contexts
        context_text = "\n\n".join(contexts)
        
        verification_prompt = ChatPromptTemplate.from_template("""
Given the following retrieved contexts and a fact, determine if the fact can be found in the contexts.

Retrieved Contexts:
{contexts}

Fact: {fact}

Is this fact present or can be inferred from the contexts?
Answer FOUND if the fact is explicitly stated or can be directly inferred.
Answer NOT_FOUND if the fact is not in the contexts.

Answer (FOUND/NOT_FOUND):""")
        
        chain = verification_prompt | self.llm | StrOutputParser()
        
        found_count = 0
        fact_results = []
        
        for fact in facts:
            result = await chain.ainvoke({
                "contexts": context_text[:4000],  # Truncate for efficiency
                "fact": fact
            })
            
            found = "FOUND" in result.upper() and "NOT" not in result.upper()
            if found:
                found_count += 1
            
            fact_results.append({"fact": fact, "found": found})
        
        score = found_count / len(facts)
        
        return MetricResult(
            metric_name="context_recall",
            score=score,
            details={
                "facts": fact_results,
                "total_facts": len(facts),
                "found_facts": found_count,
            }
        )


# =============================================================================
# Evaluator
# =============================================================================


class RAGASEvaluator:
    """
    Main evaluator class that orchestrates RAGAS evaluation.
    
    Usage:
        evaluator = RAGASEvaluator()
        report = await evaluator.evaluate_dataset(dataset)
        evaluator.print_report(report)
    """
    
    def __init__(self, llm=None):
        self.metrics = RAGASMetrics(llm=llm)
    
    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        include_recall: bool = True,
    ) -> SampleEvaluation:
        """
        Evaluate a single sample on all metrics.
        
        Args:
            sample: The evaluation sample
            include_recall: Whether to evaluate context recall (requires ground_truth)
        """
        eval_result = SampleEvaluation(sample=sample)
        
        # Faithfulness
        try:
            eval_result.faithfulness = await self.metrics.evaluate_faithfulness(
                question=sample.question,
                answer=sample.answer,
                contexts=sample.contexts,
            )
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
        
        # Answer Relevancy
        try:
            eval_result.answer_relevancy = await self.metrics.evaluate_answer_relevancy(
                question=sample.question,
                answer=sample.answer,
            )
        except Exception as e:
            logger.error(f"Answer relevancy evaluation failed: {e}")
        
        # Context Precision
        try:
            eval_result.context_precision = await self.metrics.evaluate_context_precision(
                question=sample.question,
                contexts=sample.contexts,
            )
        except Exception as e:
            logger.error(f"Context precision evaluation failed: {e}")
        
        # Context Recall (requires ground truth)
        if include_recall and sample.ground_truth:
            try:
                eval_result.context_recall = await self.metrics.evaluate_context_recall(
                    question=sample.question,
                    contexts=sample.contexts,
                    ground_truth=sample.ground_truth,
                )
            except Exception as e:
                logger.error(f"Context recall evaluation failed: {e}")
        
        return eval_result
    
    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        include_recall: bool = True,
    ) -> EvaluationReport:
        """
        Evaluate an entire dataset.
        
        Args:
            dataset: The evaluation dataset
            include_recall: Whether to evaluate context recall
        
        Returns:
            Complete evaluation report with aggregate metrics
        """
        logger.info(f"Evaluating dataset: {dataset.name} ({len(dataset.samples)} samples)")
        
        sample_evaluations = []
        
        for i, sample in enumerate(dataset.samples):
            logger.info(f"Evaluating sample {i+1}/{len(dataset.samples)}: {sample.question[:50]}...")
            eval_result = await self.evaluate_sample(sample, include_recall=include_recall)
            sample_evaluations.append(eval_result)
        
        report = EvaluationReport(
            dataset_name=dataset.name,
            timestamp=datetime.now().isoformat(),
            sample_evaluations=sample_evaluations,
        )
        report.compute_aggregates()
        
        logger.info(f"Evaluation complete. Overall score: {report.overall_score:.2f}")
        
        return report
    
    def print_report(self, report: EvaluationReport):
        """Print a formatted evaluation report."""
        print("\n" + "=" * 70)
        print(f"RAGAS EVALUATION REPORT")
        print(f"Dataset: {report.dataset_name}")
        print(f"Timestamp: {report.timestamp}")
        print("=" * 70)
        
        print("\n📊 AGGREGATE METRICS")
        print("-" * 40)
        print(f"  Faithfulness:       {report.avg_faithfulness:.3f} {'✅' if report.avg_faithfulness >= 0.7 else '⚠️'}")
        print(f"  Answer Relevancy:   {report.avg_answer_relevancy:.3f} {'✅' if report.avg_answer_relevancy >= 0.7 else '⚠️'}")
        print(f"  Context Precision:  {report.avg_context_precision:.3f} {'✅' if report.avg_context_precision >= 0.7 else '⚠️'}")
        print(f"  Context Recall:     {report.avg_context_recall:.3f} {'✅' if report.avg_context_recall >= 0.7 else '⚠️'}")
        print("-" * 40)
        print(f"  OVERALL SCORE:      {report.overall_score:.3f} {'✅' if report.overall_score >= 0.7 else '⚠️'}")
        
        print("\n🔍 ERROR ANALYSIS")
        print("-" * 40)
        if report.low_faithfulness_samples:
            print(f"  Low Faithfulness ({len(report.low_faithfulness_samples)} samples): {report.low_faithfulness_samples}")
        if report.low_relevancy_samples:
            print(f"  Low Relevancy ({len(report.low_relevancy_samples)} samples): {report.low_relevancy_samples}")
        if report.low_precision_samples:
            print(f"  Low Precision ({len(report.low_precision_samples)} samples): {report.low_precision_samples}")
        if report.low_recall_samples:
            print(f"  Low Recall ({len(report.low_recall_samples)} samples): {report.low_recall_samples}")
        
        if not any([report.low_faithfulness_samples, report.low_relevancy_samples,
                    report.low_precision_samples, report.low_recall_samples]):
            print("  No significant issues detected! 🎉")
        
        print("\n" + "=" * 70)
    
    def export_report(self, report: EvaluationReport, path: str | Path):
        """Export report to JSON file."""
        path = Path(path)
        
        # Convert to serializable format
        data = {
            "dataset_name": report.dataset_name,
            "timestamp": report.timestamp,
            "aggregate_metrics": {
                "faithfulness": report.avg_faithfulness,
                "answer_relevancy": report.avg_answer_relevancy,
                "context_precision": report.avg_context_precision,
                "context_recall": report.avg_context_recall,
                "overall_score": report.overall_score,
            },
            "error_analysis": {
                "low_faithfulness_samples": report.low_faithfulness_samples,
                "low_relevancy_samples": report.low_relevancy_samples,
                "low_precision_samples": report.low_precision_samples,
                "low_recall_samples": report.low_recall_samples,
            },
            "sample_evaluations": [],
        }
        
        for eval in report.sample_evaluations:
            sample_data = {
                "question": eval.sample.question,
                "answer": eval.sample.answer,
                "ground_truth": eval.sample.ground_truth,
                "overall_score": eval.overall_score,
                "metrics": {}
            }
            
            if eval.faithfulness:
                sample_data["metrics"]["faithfulness"] = {
                    "score": eval.faithfulness.score,
                    "details": eval.faithfulness.details,
                }
            if eval.answer_relevancy:
                sample_data["metrics"]["answer_relevancy"] = {
                    "score": eval.answer_relevancy.score,
                    "details": eval.answer_relevancy.details,
                }
            if eval.context_precision:
                sample_data["metrics"]["context_precision"] = {
                    "score": eval.context_precision.score,
                    "details": eval.context_precision.details,
                }
            if eval.context_recall:
                sample_data["metrics"]["context_recall"] = {
                    "score": eval.context_recall.score,
                    "details": eval.context_recall.details,
                }
            
            data["sample_evaluations"].append(sample_data)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Report exported to {path}")


# =============================================================================
# Factory Functions
# =============================================================================


def get_ragas_evaluator(llm=None) -> RAGASEvaluator:
    """Get a RAGAS evaluator instance."""
    return RAGASEvaluator(llm=llm)


def create_evaluation_sample(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str | None = None,
    **metadata,
) -> EvaluationSample:
    """Helper function to create an evaluation sample."""
    return EvaluationSample(
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def create_evaluation_dataset(
    name: str,
    samples: list[EvaluationSample],
    description: str = "",
) -> EvaluationDataset:
    """Helper function to create an evaluation dataset."""
    return EvaluationDataset(
        name=name,
        description=description,
        samples=samples,
    )