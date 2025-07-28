from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataConnectorError(Exception):
    pass


class ConnectionError(DataConnectorError):
    pass


class QueryExecutionError(DataConnectorError):
    pass


class DataConnector(ABC):
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.connection = None
        self._is_connected = False
        
    @abstractmethod
    def connect(self) -> None:
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        pass
    
    @abstractmethod
    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def _log_error(self, error: Exception, operation: str) -> None:
        logger.error(f"Error during {operation}: {str(error)}")
    
    def _validate_connection(self) -> None:
        if not self.is_connected:
            raise ConnectionError("Not connected to database. Call connect() first.")