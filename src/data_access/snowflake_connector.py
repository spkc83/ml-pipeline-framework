import logging
from typing import Any, Dict, Optional
import pandas as pd
import snowflake.connector
from snowflake.connector import DatabaseError, OperationalError
from snowflake.sqlalchemy import URL
import sqlalchemy
from sqlalchemy import create_engine

from .base import DataConnector, ConnectionError, QueryExecutionError

logger = logging.getLogger(__name__)


class SnowflakeConnector(DataConnector):
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.account = connection_params.get('account')
        self.username = connection_params.get('username')
        self.password = connection_params.get('password')
        self.database = connection_params.get('database')
        self.schema = connection_params.get('schema', 'PUBLIC')
        self.warehouse = connection_params.get('warehouse')
        self.role = connection_params.get('role')
        self.region = connection_params.get('region')
        self.authenticator = connection_params.get('authenticator', 'snowflake')
        self.private_key = connection_params.get('private_key')
        self.private_key_passphrase = connection_params.get('private_key_passphrase')
        self.token = connection_params.get('token')
        self.session_parameters = connection_params.get('session_parameters', {})
        self.client_session_keep_alive = connection_params.get('client_session_keep_alive', True)
        self.engine = None
        
    def connect(self) -> None:
        try:
            logger.info(f"Connecting to Snowflake account: {self.account}")
            
            conn_params = {
                'account': self.account,
                'user': self.username,
                'database': self.database,
                'schema': self.schema,
                'client_session_keep_alive': self.client_session_keep_alive,
                'session_parameters': self.session_parameters
            }
            
            if self.warehouse:
                conn_params['warehouse'] = self.warehouse
            if self.role:
                conn_params['role'] = self.role
            if self.region:
                conn_params['region'] = self.region
            
            # Handle different authentication methods
            if self.authenticator == 'externalbrowser':
                conn_params['authenticator'] = 'externalbrowser'
            elif self.authenticator == 'oauth':
                conn_params['authenticator'] = 'oauth'
                conn_params['token'] = self.token
            elif self.private_key:
                conn_params['private_key'] = self.private_key
                if self.private_key_passphrase:
                    conn_params['private_key_passphrase'] = self.private_key_passphrase
            else:
                conn_params['password'] = self.password
            
            # Create SQLAlchemy engine
            engine_url = URL(**conn_params)
            self.engine = create_engine(
                engine_url,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            
            # Also create a direct snowflake connection for non-pandas operations
            self.connection = snowflake.connector.connect(**conn_params)
            
            self._is_connected = True
            logger.info("Successfully connected to Snowflake")
            
        except (DatabaseError, OperationalError) as e:
            self._log_error(e, "Snowflake connection")
            raise ConnectionError(f"Failed to connect to Snowflake: {str(e)}")
        except Exception as e:
            self._log_error(e, "Snowflake connection")
            raise ConnectionError(f"Unexpected error connecting to Snowflake: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._validate_connection()
        
        try:
            logger.info(f"Executing Snowflake query: {query[:100]}...")
            
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if cursor.description:
                result = cursor.fetchall()
            else:
                result = cursor.rowcount
            
            cursor.close()
            
            logger.info("Query executed successfully")
            return result
            
        except DatabaseError as e:
            self._log_error(e, "query execution")
            raise QueryExecutionError(f"Failed to execute query: {str(e)}")
        except Exception as e:
            self._log_error(e, "query execution")
            raise QueryExecutionError(f"Unexpected error executing query: {str(e)}")
    
    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self._validate_connection()
        
        try:
            logger.info(f"Fetching data from Snowflake: {query[:100]}...")
            
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
            logger.info("Snowflake connection closed")
        except Exception as e:
            self._log_error(e, "connection close")
            raise ConnectionError(f"Error closing Snowflake connection: {str(e)}")