"""
Application Configuration Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Interactive Causal Policy Optimization Lab"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    # Data Storage
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    REPORTS_DIR: str = "./reports"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Session Management
    SESSION_TIMEOUT: int = 3600  # 1 hour
    MAX_SESSIONS: int = 100
    
    # Causal Learning Parameters
    DEFAULT_ALPHA: float = 0.05  # Significance level for independence tests
    DEFAULT_SCORE: str = "bic"  # Score function for structure learning
    MAX_ITERATIONS: int = 1000
    N_BOOTSTRAP_SAMPLES: int = 100
    
    # Optimization Parameters
    OPTIMIZATION_TOLERANCE: float = 1e-6
    MAX_OPTIMIZATION_ITERATIONS: int = 10000
    
    # Uncertainty Quantification
    N_POSTERIOR_SAMPLES: int = 1000
    CONFIDENCE_LEVEL: float = 0.95
    
    # Redis (for async task management)
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    
    # Random Seed for Reproducibility
    DEFAULT_RANDOM_SEED: int = 42
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.MODELS_DIR, settings.REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)