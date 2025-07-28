import logging
from typing import Any, Dict, Optional
import pandas as pd
import pymysql
from pymysql import OperationalError, DatabaseError
import sqlalchemy
from sqlalchemy import create_engine

from .base import DataConnector, ConnectionError, QueryExecutionError

logger = logging.getLogger(__name__)


class MySQLConnector(DataConnector):
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.host = connection_params.get('host', 'localhost')
        self.port = connection_params.get('port', 3306)
        self.database = connection_params.get('database')
        self.username = connection_params.get('username')
        self.password = connection_params.get('password')
        self.charset = connection_params.get('charset', 'utf8mb4')
        self.use_ssl = connection_params.get('use_ssl', False)
        self.ssl_ca = connection_params.get('ssl_ca')
        self.ssl_cert = connection_params.get('ssl_cert')
        self.ssl_key = connection_params.get('ssl_key')
        self.connect_timeout = connection_params.get('connect_timeout', 30)
        self.read_timeout = connection_params.get('read_timeout', 30)
        self.write_timeout = connection_params.get('write_timeout', 30)
        self.engine = None
        
    def connect(self) -> None:
        try:
            logger.info(f"Connecting to MySQL at {self.host}:{self.port}/{self.database}")
            
            ssl_config = {}
            if self.use_ssl:
                ssl_config = {'ssl_disabled': False}
                if self.ssl_ca:
                    ssl_config['ssl_ca'] = self.ssl_ca
                if self.ssl_cert:
                    ssl_config['ssl_cert'] = self.ssl_cert
                if self.ssl_key:
                    ssl_config['ssl_key'] = self.ssl_key
            
            connection_string = (
                f"mysql+pymysql://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
                f"?charset={self.charset}"
            )
            
            engine_args = {
                'pool_pre_ping': True,
                'pool_recycle': 3600,
                'connect_args': {
                    'connect_timeout': self.connect_timeout,
                    'read_timeout': self.read_timeout,
                    'write_timeout': self.write_timeout,
                    **ssl_config
                }
            }
            
            self.engine = create_engine(connection_string, **engine_args)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            
            # Also create a direct pymysql connection for non-pandas operations
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database,
                charset=self.charset,
                connect_timeout=self.connect_timeout,
                read_timeout=self.read_timeout,
                write_timeout=self.write_timeout,
                autocommit=False,
                **ssl_config
            )
            
            self._is_connected = True
            logger.info("Successfully connected to MySQL")
            
        except OperationalError as e:
            self._log_error(e, "MySQL connection")
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}")
        except Exception as e:
            self._log_error(e, "MySQL connection")
            raise ConnectionError(f"Unexpected error connecting to MySQL: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._validate_connection()
        
        try:
            logger.info(f"Executing MySQL query: {query[:100]}...")
            
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
            logger.info(f"Fetching data from MySQL: {query[:100]}...")
            
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
            logger.info("MySQL connection closed")
        except Exception as e:
            self._log_error(e, "connection close")
            raise ConnectionError(f"Error closing MySQL connection: {str(e)}")