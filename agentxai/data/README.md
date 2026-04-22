# data

Handles all data concerns: loading the MedQA dataset from HuggingFace, building the FAISS vector index from a PubMed subset, and defining every dataclass schema (TrajectoryEvent, AgentPlan, ToolUseEvent, MemoryDiff, AgentMessage, CausalEdge, AccountabilityReport) used throughout the system.
