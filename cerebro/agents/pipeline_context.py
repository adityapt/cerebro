"""
Pipeline Context for Agent Communication

Inspired by:
- LangGraph: Typed state with routing
- CrewAI: Structured task outputs  
- AutoGen: Message-based communication
- Semantic Kernel: Mutable context variables

This provides a standardized way for agents to communicate in the Cerebro pipeline.
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path
import uuid


class AgentStatus(str, Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentExecution(BaseModel):
    """Record of a single agent execution in the pipeline"""
    agent_name: str
    status: AgentStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class PipelineContext(BaseModel):
    """
    Context passed between agents in the Cerebro MMM pipeline.
    
    This standardized interface ensures all agents communicate consistently,
    making the orchestrator simple and extensible.
    
    Key Features:
    - ROUTING: Controls which agent runs next
    - DATA FLOW: Tracks paths and outputs
    - ERROR HANDLING: Retry logic and failure recovery
    - OBSERVABILITY: Full execution history and tracing
    - CHECKPOINTING: Resume from any stage
    - VALIDATION: Type-safe with Pydantic
    """
    
    # ============================================================================
    # IDENTITY & TRACING
    # ============================================================================
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this pipeline run"
    )
    
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Trace ID for observability (can be shared across runs)"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this pipeline run started"
    )
    
    # ============================================================================
    # ROUTING & CONTROL FLOW
    # ============================================================================
    current_agent: Optional[str] = Field(
        default=None,
        description="Currently executing agent"
    )
    
    next_agent: Optional[str] = Field(
        default="exploration",
        description="Next agent to execute (None = pipeline complete)"
    )
    
    completed_agents: List[str] = Field(
        default_factory=list,
        description="List of successfully completed agents"
    )
    
    skipped_agents: List[str] = Field(
        default_factory=list,
        description="List of skipped agents (due to failure or conditions)"
    )
    
    # ============================================================================
    # DATA FLOW
    # ============================================================================
    data_path: str = Field(
        ...,
        description="Path to current dataset (CSV)"
    )
    
    manifest_path: Optional[str] = Field(
        default=None,
        description="Path to manifest JSON (from modeling module)"
    )
    
    output_dir: str = Field(
        default=".",
        description="Directory for output files"
    )
    
    @validator('data_path')
    def data_path_must_exist(cls, v):
        """Validate that data path exists"""
        if v and not Path(v).exists():
            raise ValueError(f"Data path does not exist: {v}")
        return v
    
    # ============================================================================
    # RESULTS
    # ============================================================================
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from each agent {agent_name: result}"
    )
    
    # ============================================================================
    # ERROR HANDLING & RECOVERY
    # ============================================================================
    status: AgentStatus = Field(
        default=AgentStatus.PENDING,
        description="Current pipeline status"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if pipeline failed"
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retries for current agent"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum retries per agent"
    )
    
    # ============================================================================
    # CHECKPOINTING
    # ============================================================================
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Directory to save checkpoints"
    )
    
    resume_from: Optional[str] = Field(
        default=None,
        description="Agent to resume from (for checkpoint recovery)"
    )
    
    # ============================================================================
    # HISTORY & OBSERVABILITY
    # ============================================================================
    agent_history: List[AgentExecution] = Field(
        default_factory=list,
        description="Execution history of all agents"
    )
    
    # ============================================================================
    # METADATA
    # ============================================================================
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (spec info, user info, etc.)"
    )
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def start_agent(self, agent_name: str) -> None:
        """Mark agent as started and record in history"""
        self.current_agent = agent_name
        self.status = AgentStatus.RUNNING
        
        exec_record = AgentExecution(
            agent_name=agent_name,
            status=AgentStatus.RUNNING,
            started_at=datetime.now()
        )
        self.agent_history.append(exec_record)
    
    def mark_agent_complete(
        self, 
        agent_name: str, 
        results: Dict[str, Any],
        next_agent: Optional[str] = None
    ) -> None:
        """Mark agent as successfully completed"""
        self.completed_agents.append(agent_name)
        self.results[agent_name] = results
        self.status = AgentStatus.SUCCESS
        
        if next_agent is not None:
            self.next_agent = next_agent
        
        # Update history
        if self.agent_history and self.agent_history[-1].agent_name == agent_name:
            exec_record = self.agent_history[-1]
            exec_record.status = AgentStatus.SUCCESS
            exec_record.completed_at = datetime.now()
            exec_record.duration_ms = int(
                (exec_record.completed_at - exec_record.started_at).total_seconds() * 1000
            )
            exec_record.results = results
    
    def mark_agent_failed(
        self, 
        agent_name: str, 
        error: str, 
        skip: bool = False
    ) -> None:
        """Mark agent as failed"""
        self.status = AgentStatus.FAILED
        self.error = error
        
        if skip:
            self.skipped_agents.append(agent_name)
            self.status = AgentStatus.SKIPPED
        
        # Update history
        if self.agent_history and self.agent_history[-1].agent_name == agent_name:
            exec_record = self.agent_history[-1]
            exec_record.status = AgentStatus.FAILED if not skip else AgentStatus.SKIPPED
            exec_record.completed_at = datetime.now()
            exec_record.error = error
    
    def get_performance_summary(self) -> Dict[str, int]:
        """Get execution time per agent in milliseconds"""
        return {
            exec.agent_name: exec.duration_ms
            for exec in self.agent_history
            if exec.duration_ms is not None
        }
    
    def get_agent_result(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get results from a specific agent"""
        return self.results.get(agent_name)
    
    def should_continue(self) -> bool:
        """Check if pipeline should continue"""
        return (
            self.status != AgentStatus.FAILED and
            self.next_agent is not None
        )
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy/datetime types to native Python types for JSON serialization"""
        import numpy as np
        from datetime import datetime
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def save_checkpoint(self) -> Optional[str]:
        """Save current state to checkpoint file"""
        if not self.checkpoint_path:
            return None
        
        checkpoint_dir = Path(self.checkpoint_path)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"checkpoint_{self.run_id}.json"
        
        # Convert to dict, clean numpy types, then serialize
        data_dict = self.model_dump()
        data_dict = self._convert_numpy_types(data_dict)
        
        import json
        checkpoint_file.write_text(json.dumps(data_dict, indent=2))
        
        return str(checkpoint_file)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_file: str) -> "PipelineContext":
        """Load context from checkpoint file"""
        import json
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for logging"""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "completed_agents": self.completed_agents,
            "skipped_agents": self.skipped_agents,
            "current_agent": self.current_agent,
            "next_agent": self.next_agent,
            "data_path": self.data_path,
            "manifest_path": self.manifest_path,
            "error": self.error,
            "performance": self.get_performance_summary()
        }

