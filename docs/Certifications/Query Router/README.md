# Route Gating

# Non-Naive RAG or RAG Agents
 non-naive RAG or RAG agents,
that strategically decide when and where to access information.
This can be implemented using some combination of branching
logic, as well as some kind of predictive planning, which we don't
really cover in this course, but it's a fundamental skill to look
into and a good next step forward.

# Agent Evaluation
RAG Agent Evaluation.
In this exercise, you'll get to create and run a starter
evaluation chain to quantitatively evaluate your agent using
LLM-as-a-Judge formulations.

Judge LLM chain to evaluate
your model for its intended use case and features.
The task of a judge can vary wildly depending on what you
want to test for, but we will try formulation that asks
the question, does my RAG chain improve over a more naive approach?

Specifically, we will generate a series of synthetic question
answer pairs that combine details from randomly selected
documents in your doc store.
Then you will generate a RAG response which tries
to answer the question through its normal processes.
We will then ask our judge LLM which one is better and
impose our own assumptions or output specifications as necessary.
If we restrict the output of our judge LLM, we'll be able to coerce
the judge logic into a simple 1 for success and 0 for failure.

Then, repeating the experiment, we'll get an average percentage
of success, and can reasonably call this a metric.
And that's it, a basic evaluator chain that tests a specific
thing repeatedly.
With a little bit of creativity, you can adopt this system
to generate synthetic data based on your objective, and
can swap the formulation around to test different attributes
or output different formats.
Some common ones include cosine similarity outputs, or a list
of extractions from which you can take the length.


# Langchain Evaluator Chains and RAGAS
For inspiration, we highly recommend checking out the
LangChain Evaluator Chains and RAGAS or RAG Assessment Framework.
These are great tools for inspiration, but also should
be customized and tuned as necessary to work with your
needs and models in mind.

Also, as a next step for those interested, feel free to look
into the use of evaluator agents.
These are agent-like chains which do extra planning under
the hood to perform multi-turn conversation evaluation or
general trajectory evaluation.

These are more advanced topics which will require quite a
bit of LLM engineering to get working, but a good system
will be able to perform more rigorous evaluations and can really
help out to quantify the effective user experience in practice.

