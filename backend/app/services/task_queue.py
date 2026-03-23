"""
Task Queue Service for Asynchronous Processing
Handles long-running tasks like model training and sampling
"""
from typing import Optional, Dict, Any, Callable, Awaitable
from enum import Enum
from datetime import datetime
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """Task container"""
    def __init__(self, task_id: str, task_type: str, 
                 coro: Awaitable, metadata: Dict[str, Any] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.coro = coro
        self.metadata = metadata or {}
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.progress = 0.0
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.session_id = metadata.get("session_id") if metadata else None


class TaskQueue:
    """
    Asynchronous task queue for long-running operations
    Supports progress tracking and cancellation
    """
    
    def __init__(self, max_concurrent: int = 4):
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent
        self._queue: asyncio.Queue = None
        self._workers: list = []
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the task queue"""
        self._queue = asyncio.Queue()
        # Start worker tasks
        for i in range(self._max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        logger.info(f"Task queue initialized with {self._max_concurrent} workers")
        
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all running tasks
        for task_id, task in self._running_tasks.items():
            task.cancel()
        
        # Stop workers
        for worker in self._workers:
            worker.cancel()
        
        self._tasks.clear()
        self._running_tasks.clear()
        self._workers.clear()
        
    async def submit(self, task_type: str, coro: Awaitable, 
                    metadata: Dict[str, Any] = None) -> str:
        """
        Submit a new task to the queue
        
        Args:
            task_type: Type of task (e.g., 'structure_learning', 'optimization')
            coro: Coroutine to execute
            metadata: Additional metadata (session_id, etc.)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = Task(task_id, task_type, coro, metadata)
        
        async with self._lock:
            self._tasks[task_id] = task
        
        await self._queue.put(task)
        logger.info(f"Task {task_id} ({task_type}) submitted to queue")
        
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status"""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result if completed"""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.RUNNING:
            async_task = self._running_tasks.get(task_id)
            if async_task:
                async_task.cancel()
                task.status = TaskStatus.CANCELLED
                return True
        
        elif task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
            
        return False
    
    async def update_progress(self, task_id: str, progress: float):
        """Update task progress (0.0 to 1.0)"""
        task = self._tasks.get(task_id)
        if task:
            task.progress = min(1.0, max(0.0, progress))
    
    async def get_session_tasks(self, session_id: str) -> list:
        """Get all tasks for a session"""
        return [
            task for task in self._tasks.values()
            if task.session_id == session_id
        ]
    
    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get next task from queue
                task = await self._queue.get()
                
                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                # Update task status
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                
                logger.info(f"Worker {worker_id} processing task {task.task_id}")
                
                # Create async task
                async_task = asyncio.create_task(task.coro)
                async with self._lock:
                    self._running_tasks[task.task_id] = async_task
                
                try:
                    # Wait for completion
                    result = await async_task
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.progress = 1.0
                    logger.info(f"Task {task.task_id} completed successfully")
                    
                except asyncio.CancelledError:
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Task {task.task_id} was cancelled")
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
                    
                finally:
                    task.completed_at = datetime.utcnow()
                    async with self._lock:
                        self._running_tasks.pop(task.task_id, None)
                        
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} shutting down")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total": len(self._tasks),
            "max_concurrent": self._max_concurrent
        }


# Progress callback helper
class ProgressCallback:
    """Helper for reporting progress from within tasks"""
    
    def __init__(self, task_queue: TaskQueue, task_id: str):
        self.task_queue = task_queue
        self.task_id = task_id
    
    async def update(self, progress: float, message: str = None):
        """Update progress"""
        await self.task_queue.update_progress(self.task_id, progress)
        if message:
            logger.info(f"Task {self.task_id}: {message} ({progress*100:.1f}%)")