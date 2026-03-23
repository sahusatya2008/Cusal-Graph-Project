"""
Session Management Service
Handles user sessions, data isolation, and session storage
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json
import os
from app.config import settings


class SessionData:
    """Session data container"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.data: Dict[str, Any] = {}
        self.datasets: Dict[str, str] = {}  # dataset_id -> file_path
        self.models: Dict[str, str] = {}  # model_id -> file_path
        self.experiments: list = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "data": self.data,
            "datasets": list(self.datasets.keys()),
            "models": list(self.models.keys()),
            "experiments": self.experiments
        }
    
    def touch(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()


class SessionManager:
    """
    Manages user sessions with data isolation
    Each session has its own storage space and data
    """
    
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self._storage_path = settings.DATA_DIR
        
    async def initialize(self):
        """Initialize session manager"""
        os.makedirs(self._storage_path, exist_ok=True)
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
    async def cleanup(self):
        """Cleanup resources"""
        self._sessions.clear()
        
    async def create_session(self, session_id: str) -> SessionData:
        """Create a new session"""
        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            
            # Check max sessions
            if len(self._sessions) >= settings.MAX_SESSIONS:
                await self._evict_oldest_session()
            
            session = SessionData(session_id)
            self._sessions[session_id] = session
            
            # Create session directory
            session_dir = os.path.join(self._storage_path, session_id)
            os.makedirs(session_dir, exist_ok=True)
            os.makedirs(os.path.join(session_dir, "datasets"), exist_ok=True)
            os.makedirs(os.path.join(session_dir, "models"), exist_ok=True)
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID"""
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its data"""
        async with self._lock:
            if session_id not in self._sessions:
                return False
            
            # Remove session data
            session = self._sessions.pop(session_id)
            session_dir = os.path.join(self._storage_path, session_id)
            
            # Delete session directory
            if os.path.exists(session_dir):
                import shutil
                shutil.rmtree(session_dir)
            
            return True
    
    async def get_active_count(self) -> int:
        """Get number of active sessions"""
        return len(self._sessions)
    
    async def store_dataset(self, session_id: str, dataset_id: str, 
                           data_path: str) -> bool:
        """Store dataset reference in session"""
        session = await self.get_session(session_id)
        if not session:
            return False
        session.datasets[dataset_id] = data_path
        return True
    
    async def get_dataset_path(self, session_id: str, 
                               dataset_id: str) -> Optional[str]:
        """Get dataset path for session"""
        session = await self.get_session(session_id)
        if not session:
            return None
        return session.datasets.get(dataset_id)
    
    async def store_model(self, session_id: str, model_id: str, 
                         model_path: str) -> bool:
        """Store model reference in session"""
        session = await self.get_session(session_id)
        if not session:
            return False
        session.models[model_id] = model_path
        return True
    
    async def get_model_path(self, session_id: str, 
                            model_id: str) -> Optional[str]:
        """Get model path for session"""
        session = await self.get_session(session_id)
        if not session:
            return None
        return session.models.get(model_id)
    
    async def set_session_data(self, session_id: str, key: str, 
                               value: Any) -> bool:
        """Set arbitrary session data"""
        session = await self.get_session(session_id)
        if not session:
            return False
        session.data[key] = value
        return True
    
    async def get_session_data(self, session_id: str, 
                               key: str) -> Optional[Any]:
        """Get session data by key"""
        session = await self.get_session(session_id)
        if not session:
            return None
        return session.data.get(key)
    
    async def _evict_oldest_session(self):
        """Evict the oldest inactive session"""
        if not self._sessions:
            return
        
        oldest_id = min(
            self._sessions.keys(),
            key=lambda sid: self._sessions[sid].last_activity
        )
        await self.delete_session(oldest_id)
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired sessions"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            await self._cleanup_expired()
    
    async def _cleanup_expired(self):
        """Remove expired sessions"""
        async with self._lock:
            now = datetime.utcnow()
            timeout = timedelta(seconds=settings.SESSION_TIMEOUT)
            
            expired = [
                sid for sid, session in self._sessions.items()
                if now - session.last_activity > timeout
            ]
            
            for sid in expired:
                await self.delete_session(sid)
    
    def get_session_directory(self, session_id: str) -> str:
        """Get the storage directory for a session"""
        return os.path.join(self._storage_path, session_id)