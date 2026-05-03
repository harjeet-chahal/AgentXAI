# data

Handles all data concerns: loading the MedQA dataset from HuggingFace, building the FAISS vector index over the 18 medical textbooks, and defining every dataclass schema (TrajectoryEvent, AgentPlan, ToolUseEvent, MemoryDiff, AgentMessage, CausalEdge, AccountabilityReport) used throughout the system.
