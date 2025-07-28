import os
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from pyhive import hive
from pyhive.exc import Error as HiveError
import ssl

from .base import DataConnector, ConnectionError, QueryExecutionError

logger = logging.getLogger(__name__)


class HiveConnector(DataConnector):
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.host = connection_params.get('host', 'localhost')
        self.port = connection_params.get('port', 10000)
        self.database = connection_params.get('database', 'default')
        self.username = connection_params.get('username')
        self.password = connection_params.get('password')
        self.auth = connection_params.get('auth', 'NONE')  # NONE, KERBEROS, LDAP
        self.kerberos_service_name = connection_params.get('kerberos_service_name', 'hive')
        self.use_ssl = connection_params.get('use_ssl', False)
        self.ssl_cert = connection_params.get('ssl_cert')
        self.ssl_ca = connection_params.get('ssl_ca')
        self.ssl_key = connection_params.get('ssl_key')
        self.thrift_transport = connection_params.get('thrift_transport')
        
    def connect(self) -> None:
        try:
            logger.info(f"Connecting to Hive at {self.host}:{self.port}")
            
            auth_kwargs = {}
            if self.auth == 'KERBEROS':
                auth_kwargs['auth'] = 'KERBEROS'
                auth_kwargs['kerberos_service_name'] = self.kerberos_service_name
            elif self.auth == 'LDAP':
                auth_kwargs['auth'] = 'LDAP'
                auth_kwargs['username'] = self.username
                auth_kwargs['password'] = self.password
            elif self.username:
                auth_kwargs['username'] = self.username
                auth_kwargs['password'] = self.password
            
            ssl_context = None
            if self.use_ssl:
                ssl_context = ssl.create_default_context()
                if self.ssl_ca:
                    ssl_context.load_verify_locations(cafile=self.ssl_ca)
                if self.ssl_cert and self.ssl_key:
                    ssl_context.load_cert_chain(self.ssl_cert, self.ssl_key)
            
            self.connection = hive.Connection(
                host=self.host,
                port=self.port,
                database=self.database,
                thrift_transport=self.thrift_transport,
                **auth_kwargs
            )
            
            # Test connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            self._is_connected = True
            logger.info("Successfully connected to Hive")
            
        except HiveError as e:
            self._log_error(e, "Hive connection")
            raise ConnectionError(f"Failed to connect to Hive: {str(e)}")
        except Exception as e:
            self._log_error(e, "Hive connection")
            raise ConnectionError(f"Unexpected error connecting to Hive: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._validate_connection()
        
        try:
            logger.info(f"Executing Hive query: {query[:100]}...")
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            cursor.close()
            
            logger.info("Query executed successfully")
            return result
            
        except HiveError as e:
            self._log_error(e, "query execution")
            raise QueryExecutionError(f"Failed to execute query: {str(e)}")
        except Exception as e:
            self._log_error(e, "query execution")
            raise QueryExecutionError(f"Unexpected error executing query: {str(e)}")
    
    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self._validate_connection()
        
        try:
            logger.info(f"Fetching data from Hive: {query[:100]}...")
            
            if params:
                df = pd.read_sql(query, self.connection, params=params)
            else:
                df = pd.read_sql(query, self.connection)
            
            logger.info(f"Successfully fetched {len(df)} rows")
            return df
            
        except HiveError as e:
            self._log_error(e, "data fetching")
            raise QueryExecutionError(f"Failed to fetch data: {str(e)}")
        except Exception as e:
            self._log_error(e, "data fetching")
            raise QueryExecutionError(f"Unexpected error fetching data: {str(e)}")
    
    def close(self) -> None:
        if self.connection:
            try:
                self.connection.close()
                self._is_connected = False
                logger.info("Hive connection closed")
            except Exception as e:
                self._log_error(e, "connection close")
                raise ConnectionError(f"Error closing Hive connection: {str(e)}")