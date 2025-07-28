import logging
from typing import Any, Dict, Optional
import pandas as pd
import psycopg2
from psycopg2 import OperationalError, DatabaseError
import sqlalchemy
from sqlalchemy import create_engine

from .base import DataConnector, ConnectionError, QueryExecutionError

logger = logging.getLogger(__name__)


class PostgreSQLConnector(DataConnector):
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.host = connection_params.get('host', 'localhost')
        self.port = connection_params.get('port', 5432)
        self.database = connection_params.get('database')
        self.username = connection_params.get('username')
        self.password = connection_params.get('password')
        self.schema = connection_params.get('schema', 'public')
        self.sslmode = connection_params.get('sslmode', 'prefer')
        self.connect_timeout = connection_params.get('connect_timeout', 30)
        self.application_name = connection_params.get('application_name', 'ml-pipeline-framework')
        self.engine = None
        
    def connect(self) -> None:
        try:
            logger.info(f"Connecting to PostgreSQL at {self.host}:{self.port}/{self.database}")
            
            connection_string = (
                f"postgresql://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
                f"?sslmode={self.sslmode}&application_name={self.application_name}"
            )
            
            self.engine = create_engine(
                connection_string,
                connect_args={
                    'connect_timeout': self.connect_timeout,
                    'options': f'-csearch_path={self.schema}'
                },
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            
            # Also create a direct psycopg2 connection for non-pandas operations
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                sslmode=self.sslmode,
                connect_timeout=self.connect_timeout,
                application_name=self.application_name
            )
            
            self._is_connected = True
            logger.info("Successfully connected to PostgreSQL")
            
        except OperationalError as e:
            self._log_error(e, "PostgreSQL connection")
            raise ConnectionError(f"Failed to connect to PostgreSQL: {str(e)}")
        except Exception as e:
            self._log_error(e, "PostgreSQL connection")
            raise ConnectionError(f"Unexpected error connecting to PostgreSQL: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._validate_connection()
        
        try:
            logger.info(f"Executing PostgreSQL query: {query[:100]}...")
            
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if cursor.description:
                    result = cursor.fetchall()
                else:
                    result = cursor.rowcount
                
                self.connection.commit()
            
            logger.info("Query executed successfully")
            return result
            
        except DatabaseError as e:
            self.connection.rollback()
            self._log_error(e, "query execution")
            raise QueryExecutionError(f"Failed to execute query: {str(e)}")
        except Exception as e:
            self.connection.rollback()
            self._log_error(e, "query execution")
            raise QueryExecutionError(f"Unexpected error executing query: {str(e)}")
    
    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self._validate_connection()
        
        try:
            logger.info(f"Fetching data from PostgreSQL: {query[:100]}...")
            
            if params:
                df = pd.read_sql(query, self.engine, params=params)
            else:
                df = pd.read_sql(query, self.engine)
            
            logger.info(f"Successfully fetched {len(df)} rows")
            return df
            
        except DatabaseError as e:
            self._log_error(e, "data fetching")
            raise QueryExecutionError(f"Failed to fetch data: {str(e)}")
        except Exception as e:
            self._log_error(e, "data fetching")
            raise QueryExecutionError(f"Unexpected error fetching data: {str(e)}")
    
    def close(self) -> None:
        try:
            if self.connection:
                self.connection.close()
            if self.engine:
                self.engine.dispose()
            self._is_connected = False
            logger.info("PostgreSQL connection closed")
        except Exception as e:
            self._log_error(e, "connection close")
            raise ConnectionError(f"Error closing PostgreSQL connection: {str(e)}")