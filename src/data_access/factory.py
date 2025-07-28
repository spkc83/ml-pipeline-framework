import os
import logging
from typing import Any, Dict, Optional
from enum import Enum

from .base import DataConnector
from .hive_connector import HiveConnector
from .postgres_connector import PostgreSQLConnector
from .mysql_connector import MySQLConnector
from .snowflake_connector import SnowflakeConnector
from .redshift_connector import RedshiftConnector

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    HIVE = "hive"
    POSTGRESQL = "postgresql"
    POSTGRES = "postgres"  # Alias for postgresql
    MYSQL = "mysql"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"


class DataConnectorFactory:
    
    _connector_mapping = {
        DatabaseType.HIVE: HiveConnector,
        DatabaseType.POSTGRESQL: PostgreSQLConnector,
        DatabaseType.POSTGRES: PostgreSQLConnector,
        DatabaseType.MYSQL: MySQLConnector,
        DatabaseType.SNOWFLAKE: SnowflakeConnector,
        DatabaseType.REDSHIFT: RedshiftConnector,
    }
    
    @classmethod
    def create_connector(cls, db_type: str, connection_params: Optional[Dict[str, Any]] = None) -> DataConnector:
        """
        Create a database connector based on the database type.
        
        Args:
            db_type: Type of database (hive, postgresql, mysql, snowflake, redshift)
            connection_params: Connection parameters. If None, will try to load from environment
            
        Returns:
            DataConnector instance
            
        Raises:
            ValueError: If db_type is not supported
            KeyError: If required connection parameters are missing
        """
        try:
            database_type = DatabaseType(db_type.lower())
        except ValueError:
            supported_types = [db_type.value for db_type in DatabaseType]
            raise ValueError(f"Unsupported database type: {db_type}. Supported types: {supported_types}")
        
        if connection_params is None:
            connection_params = cls._load_connection_params_from_env(database_type)
        
        connector_class = cls._connector_mapping[database_type]
        
        logger.info(f"Creating {database_type.value} connector")
        return connector_class(connection_params)
    
    @classmethod
    def _load_connection_params_from_env(cls, db_type: DatabaseType) -> Dict[str, Any]:
        """
        Load connection parameters from environment variables.
        Environment variables should follow the pattern: {DB_TYPE}_{PARAM_NAME}
        """
        prefix = db_type.value.upper()
        
        # Common parameters for all database types
        base_params = {
            'host': os.getenv(f'{prefix}_HOST'),
            'port': cls._get_env_int(f'{prefix}_PORT'),
            'database': os.getenv(f'{prefix}_DATABASE'),
            'username': os.getenv(f'{prefix}_USERNAME') or os.getenv(f'{prefix}_USER'),
            'password': os.getenv(f'{prefix}_PASSWORD'),
        }
        
        # Remove None values
        base_params = {k: v for k, v in base_params.items() if v is not None}
        
        # Database-specific parameters
        if db_type == DatabaseType.HIVE:
            base_params.update({
                'auth': os.getenv(f'{prefix}_AUTH', 'NONE'),
                'kerberos_service_name': os.getenv(f'{prefix}_KERBEROS_SERVICE_NAME', 'hive'),
                'use_ssl': cls._get_env_bool(f'{prefix}_USE_SSL', False),
                'ssl_cert': os.getenv(f'{prefix}_SSL_CERT'),
                'ssl_ca': os.getenv(f'{prefix}_SSL_CA'),
                'ssl_key': os.getenv(f'{prefix}_SSL_KEY'),
                'thrift_transport': os.getenv(f'{prefix}_THRIFT_TRANSPORT'),
            })
        
        elif db_type in [DatabaseType.POSTGRESQL, DatabaseType.POSTGRES]:
            base_params.update({
                'schema': os.getenv(f'{prefix}_SCHEMA', 'public'),
                'sslmode': os.getenv(f'{prefix}_SSLMODE', 'prefer'),
                'connect_timeout': cls._get_env_int(f'{prefix}_CONNECT_TIMEOUT', 30),
                'application_name': os.getenv(f'{prefix}_APPLICATION_NAME', 'ml-pipeline-framework'),
            })
        
        elif db_type == DatabaseType.MYSQL:
            base_params.update({
                'charset': os.getenv(f'{prefix}_CHARSET', 'utf8mb4'),
                'use_ssl': cls._get_env_bool(f'{prefix}_USE_SSL', False),
                'ssl_ca': os.getenv(f'{prefix}_SSL_CA'),
                'ssl_cert': os.getenv(f'{prefix}_SSL_CERT'),
                'ssl_key': os.getenv(f'{prefix}_SSL_KEY'),
                'connect_timeout': cls._get_env_int(f'{prefix}_CONNECT_TIMEOUT', 30),
            })
        
        elif db_type == DatabaseType.SNOWFLAKE:
            base_params.update({
                'account': os.getenv(f'{prefix}_ACCOUNT'),
                'schema': os.getenv(f'{prefix}_SCHEMA', 'PUBLIC'),
                'warehouse': os.getenv(f'{prefix}_WAREHOUSE'),
                'role': os.getenv(f'{prefix}_ROLE'),
                'region': os.getenv(f'{prefix}_REGION'),
                'authenticator': os.getenv(f'{prefix}_AUTHENTICATOR', 'snowflake'),
                'private_key': os.getenv(f'{prefix}_PRIVATE_KEY'),
                'private_key_passphrase': os.getenv(f'{prefix}_PRIVATE_KEY_PASSPHRASE'),
                'token': os.getenv(f'{prefix}_TOKEN'),
            })
            
            # Handle session parameters
            session_params = {}
            for key, value in os.environ.items():
                if key.startswith(f'{prefix}_SESSION_'):
                    param_name = key.replace(f'{prefix}_SESSION_', '').lower()
                    session_params[param_name] = value
            if session_params:
                base_params['session_parameters'] = session_params
        
        elif db_type == DatabaseType.REDSHIFT:
            base_params.update({
                'schema': os.getenv(f'{prefix}_SCHEMA', 'public'),
                'sslmode': os.getenv(f'{prefix}_SSLMODE', 'require'),
                'connect_timeout': cls._get_env_int(f'{prefix}_CONNECT_TIMEOUT', 30),
                'application_name': os.getenv(f'{prefix}_APPLICATION_NAME', 'ml-pipeline-framework'),
                'cluster_identifier': os.getenv(f'{prefix}_CLUSTER_IDENTIFIER'),
                'db_user': os.getenv(f'{prefix}_DB_USER'),
                'auto_create': cls._get_env_bool(f'{prefix}_AUTO_CREATE', False),
                'region': os.getenv(f'{prefix}_REGION'),
                'iam_role': os.getenv(f'{prefix}_IAM_ROLE'),
            })
            
            # Handle db_groups
            db_groups = os.getenv(f'{prefix}_DB_GROUPS')
            if db_groups:
                base_params['db_groups'] = db_groups.split(',')
        
        # Remove None values again after adding specific parameters
        base_params = {k: v for k, v in base_params.items() if v is not None}
        
        # Validate required parameters
        cls._validate_required_params(db_type, base_params)
        
        return base_params
    
    @classmethod
    def _validate_required_params(cls, db_type: DatabaseType, params: Dict[str, Any]) -> None:
        """Validate that required parameters are present for each database type."""
        required_params = {
            DatabaseType.HIVE: ['host'],
            DatabaseType.POSTGRESQL: ['host', 'database', 'username'],
            DatabaseType.POSTGRES: ['host', 'database', 'username'],
            DatabaseType.MYSQL: ['host', 'database', 'username'],
            DatabaseType.SNOWFLAKE: ['account', 'username', 'database'],
            DatabaseType.REDSHIFT: ['host', 'database', 'username'],
        }
        
        missing_params = []
        for param in required_params.get(db_type, []):
            if param not in params or params[param] is None:
                missing_params.append(param)
        
        if missing_params:
            prefix = db_type.value.upper()
            env_vars = [f'{prefix}_{param.upper()}' for param in missing_params]
            raise KeyError(
                f"Missing required connection parameters for {db_type.value}: {missing_params}. "
                f"Set environment variables: {env_vars}"
            )
    
    @staticmethod
    def _get_env_int(env_var: str, default: Optional[int] = None) -> Optional[int]:
        """Get environment variable as integer."""
        value = os.getenv(env_var)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {env_var}: {value}")
            return default
    
    @staticmethod
    def _get_env_bool(env_var: str, default: bool = False) -> bool:
        """Get environment variable as boolean."""
        value = os.getenv(env_var, '').lower()
        return value in ('true', '1', 'yes', 'on')
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> DataConnector:
        """
        Create a connector from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and connection parameters
            
        Returns:
            DataConnector instance
        """
        if 'type' not in config:
            raise KeyError("Configuration must include 'type' field")
        
        db_type = config.pop('type')
        connection_params = config
        
        return cls.create_connector(db_type, connection_params)