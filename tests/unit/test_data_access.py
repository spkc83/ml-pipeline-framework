"""
Unit tests for data access modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
import psycopg2
import pymysql
from unittest.mock import Mock

from src.data_access.base import DataConnector
from src.data_access.factory import DataConnectorFactory
from src.data_access.postgres_connector import PostgreSQLConnector
from src.data_access.mysql_connector import MySQLConnector
from src.data_access.hive_connector import HiveConnector
from src.data_access.snowflake_connector import SnowflakeConnector
from src.data_access.redshift_connector import RedshiftConnector


class TestDataConnectorBase:
    """Test cases for the base DataConnector class."""
    
    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        connector = DataConnector()
        
        with pytest.raises(NotImplementedError):
            connector.connect()
        
        with pytest.raises(NotImplementedError):
            connector.execute_query("SELECT 1")
        
        with pytest.raises(NotImplementedError):
            connector.fetch_data("SELECT 1")
        
        with pytest.raises(NotImplementedError):
            connector.close()


class TestDataConnectorFactory:
    """Test cases for DataConnectorFactory."""
    
    def test_create_postgresql_connector(self, postgres_config):
        """Test creating PostgreSQL connector."""
        connector = DataConnectorFactory.create_connector('postgresql', postgres_config)
        assert isinstance(connector, PostgreSQLConnector)
        assert connector.connection_params == postgres_config
    
    def test_create_mysql_connector(self, mysql_config):
        """Test creating MySQL connector."""
        connector = DataConnectorFactory.create_connector('mysql', mysql_config)
        assert isinstance(connector, MySQLConnector)
        assert connector.connection_params == mysql_config
    
    def test_create_hive_connector(self):
        """Test creating Hive connector."""
        config = {'host': 'localhost', 'port': 10000}
        connector = DataConnectorFactory.create_connector('hive', config)
        assert isinstance(connector, HiveConnector)
    
    def test_create_snowflake_connector(self, snowflake_config):
        """Test creating Snowflake connector."""
        connector = DataConnectorFactory.create_connector('snowflake', snowflake_config)
        assert isinstance(connector, SnowflakeConnector)
    
    def test_create_redshift_connector(self, postgres_config):
        """Test creating Redshift connector."""
        connector = DataConnectorFactory.create_connector('redshift', postgres_config)
        assert isinstance(connector, RedshiftConnector)
    
    def test_unsupported_connector_type(self):
        """Test error for unsupported connector type."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            DataConnectorFactory.create_connector('unsupported', {})
    
    def test_missing_connection_params(self):
        """Test error for missing connection parameters."""
        with pytest.raises(ValueError, match="Connection parameters are required"):
            DataConnectorFactory.create_connector('postgresql', None)


class TestPostgreSQLConnector:
    """Test cases for PostgreSQLConnector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }
        self.connector = PostgreSQLConnector(self.config)
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_successful_connection(self, mock_connect):
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        
        mock_connect.assert_called_once_with(
            host='localhost',
            port=5432,
            database='test_db',
            user='test_user',
            password='test_password'
        )
        assert self.connector.connection == mock_conn
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_connection_failure(self, mock_connect):
        """Test connection failure handling."""
        mock_connect.side_effect = psycopg2.Error("Connection failed")
        
        with pytest.raises(Exception, match="Failed to connect to PostgreSQL"):
            self.connector.connect()
    
    def test_execute_query_without_connection(self):
        """Test executing query without connection."""
        with pytest.raises(Exception, match="No active connection"):
            self.connector.execute_query("SELECT 1")
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_execute_query_success(self, mock_connect):
        """Test successful query execution."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        result = self.connector.execute_query("SELECT * FROM test_table")
        
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table")
        assert result == mock_cursor
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_fetch_data_success(self, mock_connect):
        """Test successful data fetching."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, 'value1', 10.5),
            (2, 'value2', 20.5)
        ]
        mock_cursor.description = [
            ('id',), ('name',), ('value',)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        df = self.connector.fetch_data("SELECT * FROM test_table")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'value']
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 1] == 'value1'
        assert df.iloc[0, 2] == 10.5
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_fetch_empty_result(self, mock_connect):
        """Test fetching empty result set."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [('id',), ('name',)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        df = self.connector.fetch_data("SELECT * FROM empty_table")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ['id', 'name']
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_close_connection(self, mock_connect):
        """Test closing connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        self.connector.close()
        
        mock_conn.close.assert_called_once()
        assert self.connector.connection is None
    
    def test_close_without_connection(self):
        """Test closing without active connection."""
        # Should not raise an exception
        self.connector.close()


class TestMySQLConnector:
    """Test cases for MySQLConnector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }
        self.connector = MySQLConnector(self.config)
    
    @patch('src.data_access.mysql_connector.pymysql.connect')
    def test_successful_connection(self, mock_connect):
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        
        mock_connect.assert_called_once_with(
            host='localhost',
            port=3306,
            database='test_db',
            user='test_user',
            password='test_password',
            cursorclass=pymysql.cursors.DictCursor
        )
        assert self.connector.connection == mock_conn
    
    @patch('src.data_access.mysql_connector.pymysql.connect')
    def test_connection_failure(self, mock_connect):
        """Test connection failure handling."""
        mock_connect.side_effect = pymysql.Error("Connection failed")
        
        with pytest.raises(Exception, match="Failed to connect to MySQL"):
            self.connector.connect()
    
    @patch('src.data_access.mysql_connector.pymysql.connect')
    def test_fetch_data_with_dict_cursor(self, mock_connect):
        """Test data fetching with dictionary cursor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'name': 'value1', 'value': 10.5},
            {'id': 2, 'name': 'value2', 'value': 20.5}
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        df = self.connector.fetch_data("SELECT * FROM test_table")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'value']


class TestHiveConnector:
    """Test cases for HiveConnector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'localhost',
            'port': 10000,
            'database': 'default'
        }
        self.connector = HiveConnector(self.config)
    
    @patch('src.data_access.hive_connector.hive.Connection')
    def test_successful_connection(self, mock_connection):
        """Test successful Hive connection."""
        mock_conn = MagicMock()
        mock_connection.return_value = mock_conn
        
        self.connector.connect()
        
        mock_connection.assert_called_once_with(
            host='localhost',
            port=10000,
            database='default'
        )
        assert self.connector.connection == mock_conn
    
    @patch('src.data_access.hive_connector.hive.Connection')
    def test_connection_failure(self, mock_connection):
        """Test Hive connection failure."""
        mock_connection.side_effect = Exception("Hive connection failed")
        
        with pytest.raises(Exception, match="Failed to connect to Hive"):
            self.connector.connect()


class TestSnowflakeConnector:
    """Test cases for SnowflakeConnector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'account': 'test_account',
            'warehouse': 'test_warehouse',
            'database': 'test_db',
            'schema': 'test_schema',
            'username': 'test_user',
            'password': 'test_password'
        }
        self.connector = SnowflakeConnector(self.config)
    
    @patch('src.data_access.snowflake_connector.snowflake.connector.connect')
    def test_successful_connection(self, mock_connect):
        """Test successful Snowflake connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        
        mock_connect.assert_called_once_with(
            account='test_account',
            warehouse='test_warehouse',
            database='test_db',
            schema='test_schema',
            user='test_user',
            password='test_password'
        )
        assert self.connector.connection == mock_conn
    
    @patch('src.data_access.snowflake_connector.snowflake.connector.connect')
    def test_connection_failure(self, mock_connect):
        """Test Snowflake connection failure."""
        mock_connect.side_effect = Exception("Snowflake connection failed")
        
        with pytest.raises(Exception, match="Failed to connect to Snowflake"):
            self.connector.connect()


class TestRedshiftConnector:
    """Test cases for RedshiftConnector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'test-cluster.region.redshift.amazonaws.com',
            'port': 5439,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }
        self.connector = RedshiftConnector(self.config)
    
    @patch('src.data_access.redshift_connector.psycopg2.connect')
    def test_successful_connection(self, mock_connect):
        """Test successful Redshift connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        self.connector.connect()
        
        mock_connect.assert_called_once_with(
            host='test-cluster.region.redshift.amazonaws.com',
            port=5439,
            database='test_db',
            user='test_user',
            password='test_password'
        )
        assert self.connector.connection == mock_conn
    
    @patch('src.data_access.redshift_connector.psycopg2.connect')
    def test_connection_failure(self, mock_connect):
        """Test Redshift connection failure."""
        mock_connect.side_effect = psycopg2.Error("Redshift connection failed")
        
        with pytest.raises(Exception, match="Failed to connect to Redshift"):
            self.connector.connect()


class TestConnectionPooling:
    """Test cases for connection pooling functionality."""
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_connection_reuse(self, mock_connect):
        """Test that connections are reused properly."""
        config = {
            'host': 'localhost',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }
        
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        connector = PostgreSQLConnector(config)
        
        # First connection
        connector.connect()
        first_connection = connector.connection
        
        # Second call should reuse connection
        connector.connect()
        second_connection = connector.connection
        
        assert first_connection == second_connection
        assert mock_connect.call_count == 1


class TestErrorHandling:
    """Test cases for error handling in data connectors."""
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_query_execution_error(self, mock_connect):
        """Test handling of query execution errors."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = psycopg2.Error("Query execution failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        connector = PostgreSQLConnector({
            'host': 'localhost',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        })
        
        connector.connect()
        
        with pytest.raises(Exception, match="Query execution failed"):
            connector.execute_query("INVALID SQL")
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_fetch_data_error(self, mock_connect):
        """Test handling of data fetching errors."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.side_effect = psycopg2.Error("Fetch failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        connector = PostgreSQLConnector({
            'host': 'localhost',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        })
        
        connector.connect()
        
        with pytest.raises(Exception, match="Failed to fetch data"):
            connector.fetch_data("SELECT * FROM test_table")


class TestDataTypeHandling:
    """Test cases for handling different data types."""
    
    @patch('src.data_access.postgres_connector.psycopg2.connect')
    def test_various_data_types(self, mock_connect):
        """Test handling of various PostgreSQL data types."""
        from datetime import datetime, date
        import decimal
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        # Mock data with various types
        mock_cursor.fetchall.return_value = [
            (1, 'text_value', 123.45, True, datetime(2023, 1, 1, 12, 0, 0), 
             date(2023, 1, 1), decimal.Decimal('999.99'), None)
        ]
        mock_cursor.description = [
            ('id',), ('text_col',), ('float_col',), ('bool_col',), 
            ('datetime_col',), ('date_col',), ('decimal_col',), ('null_col',)
        ]
        
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        connector = PostgreSQLConnector({
            'host': 'localhost',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        })
        
        connector.connect()
        df = connector.fetch_data("SELECT * FROM test_table")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0, 0] == 1  # id
        assert df.iloc[0, 1] == 'text_value'  # text
        assert df.iloc[0, 2] == 123.45  # float
        assert df.iloc[0, 3] == True  # boolean
        assert pd.isna(df.iloc[0, 7])  # null value