# Data Access API

The data access module provides connectors for various data sources with a unified interface for data retrieval and management.

## Classes

### BaseConnector

```python
class BaseConnector(ABC)
```

Abstract base class for all data connectors providing a unified interface for data access.

This class defines the contract that all data connectors must implement, ensuring consistency across different data sources. It provides common functionality for connection management, query execution, and data validation.

**Attributes:**
- `connection_config` (Dict[str, Any]): Connection configuration parameters
- `is_connected` (bool): Current connection status
- `last_query_time` (float): Timestamp of last query execution
- `query_cache` (Dict): Cache for frequently executed queries

**Example:**
```python
from ml_pipeline_framework.data_access import PostgresConnector

connector = PostgresConnector(
    connection_string="postgresql://user:pass@host:5432/db",
    pool_size=10
)

data = connector.query("SELECT * FROM customers WHERE active = true")
```

#### Methods

##### `connect()`

Establish connection to the data source.

**Raises:**
- `ConnectionError`: If connection cannot be established
- `AuthenticationError`: If credentials are invalid

##### `disconnect()`

Close connection to the data source.

##### `query(sql, parameters=None)`

Execute SQL query and return results.

**Args:**
- `sql` (str): SQL query string
- `parameters` (Dict, optional): Query parameters for prepared statements

**Returns:**
- `pd.DataFrame`: Query results as pandas DataFrame

**Raises:**
- `DatabaseError`: If query execution fails
- `ValidationError`: If query validation fails

##### `execute(sql, parameters=None)`

Execute SQL statement without returning results.

**Args:**
- `sql` (str): SQL statement
- `parameters` (Dict, optional): Statement parameters

**Returns:**
- `int`: Number of affected rows

##### `get_table_schema(table_name)`

Get schema information for a table.

**Args:**
- `table_name` (str): Name of the table

**Returns:**
- `Dict[str, Any]`: Table schema information

##### `validate_connection()`

Validate current connection status.

**Returns:**
- `bool`: True if connection is valid

---

### ConnectorFactory

```python
class ConnectorFactory
```

Factory class for creating data connector instances based on configuration.

The factory provides a centralized way to create connectors with proper configuration validation and dependency checking. It supports dynamic connector registration and automatic credential management.

**Example:**
```python
from ml_pipeline_framework.data_access import ConnectorFactory

# Create connector from configuration
config = {
    'type': 'postgres',
    'connection_string': 'postgresql://...',
    'pool_size': 10
}

connector = ConnectorFactory.create_connector(config)
```

#### Methods

##### `create_connector(config)`

Create connector instance based on configuration.

**Args:**
- `config` (Dict[str, Any]): Connector configuration

**Returns:**
- `BaseConnector`: Configured connector instance

**Raises:**
- `ValueError`: If connector type is not supported
- `ConfigurationError`: If configuration is invalid

##### `register_connector(connector_type, connector_class)`

Register a new connector type.

**Args:**
- `connector_type` (str): Type identifier for the connector
- `connector_class` (Type[BaseConnector]): Connector class to register

##### `get_supported_types()`

Get list of supported connector types.

**Returns:**
- `List[str]`: List of supported connector type identifiers

---

### PostgresConnector

```python
class PostgresConnector(BaseConnector)
```

PostgreSQL database connector with connection pooling and advanced query features.

Provides optimized access to PostgreSQL databases with support for connection pooling, prepared statements, bulk operations, and transaction management.

**Attributes:**
- `connection_pool` (psycopg2.pool.ThreadedConnectionPool): Connection pool for concurrent access
- `default_schema` (str): Default schema for queries
- `query_timeout` (int): Default query timeout in seconds

**Example:**
```python
connector = PostgresConnector(
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    pool_size=20,
    query_timeout=300
)

# Execute query with parameters
data = connector.query(
    "SELECT * FROM sales WHERE date >= %(start_date)s",
    parameters={'start_date': '2023-01-01'}
)

# Bulk insert
connector.bulk_insert('staging_table', dataframe)
```

#### Methods

##### `__init__(connection_string, pool_size=10, query_timeout=300, **kwargs)`

Initialize PostgreSQL connector.

**Args:**
- `connection_string` (str): PostgreSQL connection string
- `pool_size` (int): Maximum connections in pool. Defaults to 10
- `query_timeout` (int): Query timeout in seconds. Defaults to 300
- `**kwargs`: Additional connection parameters

##### `bulk_insert(table_name, data, batch_size=1000, on_conflict='raise')`

Perform bulk insert operation.

**Args:**
- `table_name` (str): Target table name
- `data` (pd.DataFrame): Data to insert
- `batch_size` (int): Batch size for insertion. Defaults to 1000
- `on_conflict` (str): Conflict resolution strategy. Defaults to 'raise'

**Returns:**
- `int`: Number of rows inserted

##### `copy_from_csv(table_name, csv_path, delimiter=',', header=True)`

Copy data from CSV file to table.

**Args:**
- `table_name` (str): Target table name
- `csv_path` (Union[str, Path]): Path to CSV file
- `delimiter` (str): CSV delimiter. Defaults to ','
- `header` (bool): Whether CSV has header row. Defaults to True

---

### SnowflakeConnector

```python
class SnowflakeConnector(BaseConnector)
```

Snowflake data warehouse connector with optimized query execution and warehouse management.

Provides enterprise-grade access to Snowflake with support for warehouse scaling, query optimization, and efficient data transfer using Snowflake's native capabilities.

**Attributes:**
- `warehouse` (str): Active Snowflake warehouse
- `database` (str): Active database
- `schema` (str): Active schema
- `role` (str): Active role

**Example:**
```python
connector = SnowflakeConnector(
    account='myaccount.snowflakecomputing.com',
    user='myuser',
    password='mypass',
    warehouse='COMPUTE_WH',
    database='ANALYTICS_DB',
    schema='PUBLIC'
)

# Execute with warehouse scaling
data = connector.query_with_scaling(
    sql="SELECT * FROM large_table",
    warehouse_size='LARGE'
)
```

#### Methods

##### `__init__(account, user, password, warehouse, database, schema, role=None, **kwargs)`

Initialize Snowflake connector.

**Args:**
- `account` (str): Snowflake account identifier
- `user` (str): Username for authentication
- `password` (str): Password for authentication
- `warehouse` (str): Default warehouse name
- `database` (str): Default database name
- `schema` (str): Default schema name
- `role` (str, optional): Role to assume
- `**kwargs`: Additional connection parameters

##### `query_with_scaling(sql, warehouse_size=None, auto_suspend=True)`

Execute query with automatic warehouse scaling.

**Args:**
- `sql` (str): SQL query to execute
- `warehouse_size` (str, optional): Warehouse size for query execution
- `auto_suspend` (bool): Auto-suspend warehouse after query. Defaults to True

**Returns:**
- `pd.DataFrame`: Query results

##### `upload_dataframe(df, table_name, if_exists='append', create_table=False)`

Upload DataFrame to Snowflake table.

**Args:**
- `df` (pd.DataFrame): DataFrame to upload
- `table_name` (str): Target table name
- `if_exists` (str): Action if table exists. Defaults to 'append'
- `create_table` (bool): Create table if not exists. Defaults to False

---

### RedshiftConnector

```python
class RedshiftConnector(BaseConnector)
```

Amazon Redshift data warehouse connector with optimized column-oriented query execution.

Provides high-performance access to Amazon Redshift with support for COPY operations, UNLOAD operations, and query optimization for analytical workloads.

**Example:**
```python
connector = RedshiftConnector(
    host='redshift-cluster.amazonaws.com',
    port=5439,
    database='analytics',
    user='analyst',
    password='password'
)

# Optimized column-oriented query
data = connector.query_columnar(
    "SELECT customer_id, SUM(amount) FROM sales GROUP BY customer_id"
)
```

#### Methods

##### `copy_from_s3(table_name, s3_path, iam_role, file_format='CSV')`

Copy data from S3 to Redshift table.

**Args:**
- `table_name` (str): Target table name
- `s3_path` (str): S3 path to data files
- `iam_role` (str): IAM role ARN for S3 access
- `file_format` (str): Data file format. Defaults to 'CSV'

##### `unload_to_s3(query, s3_path, iam_role, options=None)`

Unload query results to S3.

**Args:**
- `query` (str): Query to execute and unload
- `s3_path` (str): Target S3 path
- `iam_role` (str): IAM role ARN for S3 access
- `options` (Dict, optional): Additional UNLOAD options

---

### HiveConnector

```python
class HiveConnector(BaseConnector)
```

Apache Hive connector for big data analytics with support for partitioned tables and complex data types.

Provides access to Hive data warehouses with optimized query execution for large-scale analytics and support for Hadoop ecosystem integration.

**Example:**
```python
connector = HiveConnector(
    host='hive-server.company.com',
    port=10000,
    database='analytics',
    auth='KERBEROS'
)

# Query partitioned table
data = connector.query_partitioned(
    table='sales_data',
    partitions={'year': 2023, 'month': 12}
)
```

#### Methods

##### `query_partitioned(table, partitions, columns=None)`

Query specific partitions of a partitioned table.

**Args:**
- `table` (str): Table name
- `partitions` (Dict[str, Any]): Partition filter conditions
- `columns` (List[str], optional): Columns to select

**Returns:**
- `pd.DataFrame`: Query results

##### `get_partitions(table_name)`

Get available partitions for a table.

**Args:**
- `table_name` (str): Table name

**Returns:**
- `List[Dict[str, Any]]`: List of partition information

---

### MySQLConnector

```python
class MySQLConnector(BaseConnector)
```

MySQL database connector with optimized query execution and connection management.

Provides efficient access to MySQL databases with support for read replicas, connection pooling, and optimized data transfer operations.

**Example:**
```python
connector = MySQLConnector(
    host='mysql.company.com',
    port=3306,
    database='application_db',
    user='app_user',
    password='app_password',
    read_replica_hosts=['replica1.company.com', 'replica2.company.com']
)

# Use read replica for analytics query
data = connector.query_replica(
    "SELECT * FROM user_events WHERE date >= '2023-01-01'"
)
```

#### Methods

##### `query_replica(sql, prefer_replica=True)`

Execute read-only query using read replica if available.

**Args:**
- `sql` (str): Read-only SQL query
- `prefer_replica` (bool): Prefer replica over master. Defaults to True

**Returns:**
- `pd.DataFrame`: Query results

##### `get_replica_lag()`

Get replication lag for read replicas.

**Returns:**
- `Dict[str, float]`: Lag in seconds for each replica

## Configuration Examples

### PostgreSQL Configuration

```yaml
data_source:
  type: "postgres"
  connection_string: "postgresql://user:password@host:5432/database"
  pool_size: 20
  query_timeout: 600
  ssl_mode: "require"
  application_name: "ml_pipeline"
```

### Snowflake Configuration

```yaml
data_source:
  type: "snowflake"
  account: "myaccount.snowflakecomputing.com"
  user: "analytics_user"
  password: "${SNOWFLAKE_PASSWORD}"
  warehouse: "ANALYTICS_WH"
  database: "PROD_DATA"
  schema: "ML_FEATURES"
  role: "ML_ANALYST"
  warehouse_size: "MEDIUM"
  auto_suspend: true
```

### Redshift Configuration

```yaml
data_source:
  type: "redshift"
  host: "redshift-cluster.amazonaws.com"
  port: 5439
  database: "analytics"
  user: "ml_user"
  password: "${REDSHIFT_PASSWORD}"
  ssl: true
  iam_role: "arn:aws:iam::123456789012:role/RedshiftS3Access"
```

## Error Handling

All connectors implement comprehensive error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **Query Errors**: Detailed error messages with query context
- **Timeout Errors**: Configurable timeouts with graceful handling
- **Authentication Errors**: Clear error messages for credential issues

## Performance Optimization

- **Connection Pooling**: Efficient connection reuse
- **Query Caching**: Intelligent caching of repeated queries
- **Batch Operations**: Optimized bulk data operations
- **Parallel Processing**: Multi-threaded query execution where supported
- **Memory Management**: Streaming results for large datasets

## Security Features

- **Credential Management**: Secure credential storage and rotation
- **SSL/TLS Support**: Encrypted connections for all connectors
- **Query Validation**: SQL injection prevention
- **Access Control**: Role-based access control integration
- **Audit Logging**: Comprehensive query and access logging