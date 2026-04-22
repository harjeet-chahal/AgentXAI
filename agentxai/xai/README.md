# xai

The XAI runtime layer — a set of observer classes (hooked via LangChain callbacks) that run in parallel to the agent pipeline and log all 7 explainability pillars: trajectory events, plan tracking, tool provenance, memory diffs, inter-agent messages, the causal DAG, the accountability report, and the counterfactual engine that measures causal impact via perturbation re-runs.
