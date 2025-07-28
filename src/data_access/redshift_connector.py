import logging
from typing import Any, Dict, Optional
import pandas as pd
import psycopg2
from psycopg2 import OperationalError, DatabaseError
import sqlalchemy
from sqlalchemy import create_engine

from .base import DataConnector, ConnectionError, QueryExecutionError

logger = logging.getLogger(__name__)


class RedshiftConnector(DataConnector):
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.host = connection_params.get('host')
        self.port = connection_params.get('port', 5439)
        self.database = connection_params.get('database')
        self.username = connection_params.get('username')
        self.password = connection_params.get('password')
        self.schema = connection_params.get('schema', 'public')
        self.sslmode = connection_params.get('sslmode', 'require')
        self.connect_timeout = connection_params.get('connect_timeout', 30)
        self.application_name = connection_params.get('application_name', 'ml-pipeline-framework')
        self.cluster_identifier = connection_params.get('cluster_identifier')
        self.db_user = connection_params.get('db_user')
        self.auto_create = connection_params.get('auto_create', False)
        self.db_groups = connection_params.get('db_groups', [])
        self.region = connection_params.get('region')
        self.iam_role = connection_params.get('iam_role')
        self.engine = None
        
    def connect(self) -> None:
        try:
            logger.info(f"Connecting to Redshift at {self.host}:{self.port}/{self.database}")
            
            # Build connection string based on authentication method
            if self.iam_role and self.cluster_identifier:
                # Use IAM role authentication
                connection_string = (
                    f"redshift+psycopg2://{self.username}:{self.password}@"
                    f"{self.host}:{self.port}/{self.database}"
                    f"?sslmode={self.sslmode}&application_name={self.application_name}"
                )
            else:
                # Use username/password authentication
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
                pool_recycle=3600,
                # Redshift-specific optimizations
                execution_options={
                    'autocommit': True,
                    'isolation_level': 'AUTOCOMMIT'
                }
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
            
            # Set autocommit for Redshift
            self.connection.autocommit = True
            
            self._is_connected = True
            logger.info("Successfully connected to Redshift")
            
        except OperationalError as e:
            self._log_error(e, "Redshift connection")
            raise ConnectionError(f"Failed to connect to Redshift: {str(e)}")
        except Exception as e:
            self._log_error(e, "Redshift connection")
            raise ConnectionError(f"Unexpected error connecting to Redshift: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._validate_connection()
        
        try:
            logger.info(f"Executing Redshift query: {query[:100]}...")
            
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if cursor.description:
                    result = cursor.fetchall()
                else:
                    result = cursor.rowcount
            
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
            logger.info(f"Fetching data from Redshift: {query[:100]}...")
            
            # Use chunksize for large datasets to avoid memory issues
            chunksize = self.connection_params.get('chunksize', 10000)
            
            if params:
                df = pd.read_sql(query, self.engine, params=params, chunksize=chunksize)
            else:
                df = pd.read_sql(query, self.engine, chunksize=chunksize)
            
            # If chunksize is specified, concatenate chunks
            if isinstance(df, pd.io.sql.DataFrameGroupBy):
                df = pd.concat(df, ignore_index=True)
            
            logger.info(f"Successfully fetched {len(df)} rows")
            return df
            
        except DatabaseError as e:
            self._log_error(e, "data fetching")
            raise QueryExecutionError(f"Failed to fetch data: {str(e)}")
        except Exception as e:
            self._log_error(e, "data fetching")
            raise QueryExecutionError(f"Unexpected error fetching data: {str(e)}")
    
    def bulk_load_from_s3(self, table_name: str, s3_path: str, 
                         credentials: str, copy_options: str = "") -> None:
        """
        Perform bulk load from S3 using COPY command - Redshift specific feature
        """
        self._validate_connection()
        
        try:
            copy_query = f"""
                COPY {self.schema}.{table_name}
                FROM '{s3_path}'
                CREDENTIALS '{credentials}'
                {copy_options}
            """
            
            logger.info(f"Executing COPY command for table {table_name}")
            self.execute_query(copy_query)
            logger.info(f"Successfully loaded data into {table_name}")
            
        except Exception as e:
            self._log_error(e, "bulk load from S3")
            raise QueryExecutionError(f"Failed to bulk load from S3: {str(e)}")
    
    def close(self) -> None:
        try:
            if self.connection:
                self.connection.close()
            if self.engine:
                self.engine.dispose()
            self._is_connected = False
            logger.info("Redshift connection closed")
        except Exception as e:
            self._log_error(e, "connection close")
            raise ConnectionError(f"Error closing Redshift connection: {str(e)}")