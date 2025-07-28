"""
Spark integration tests for the ML Pipeline Framework.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Skip all tests if PySpark is not available
pyspark = pytest.importorskip("pyspark")
pytest_spark = pytest.importorskip("pyspark.testing")

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.pipeline_orchestrator import PipelineOrchestrator


@pytest.fixture(scope="module")
def spark_session():
    """Create Spark session for testing."""
    spark = SparkSession.builder \
        .appName("MLPipelineFrameworkTests") \
        .master("local[2]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    yield spark
    
    spark.stop()


@pytest.fixture
def spark_test_data(spark_session):
    """Create test data as Spark DataFrame."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    for i in range(n_samples):
        row = {
            'feature_1': float(np.random.normal(0, 1)),
            'feature_2': float(np.random.uniform(0, 10)),
            'feature_3': int(np.random.randint(1, 5)),
            'category': np.random.choice(['A', 'B', 'C']),
            'target': int(np.random.choice([0, 1]))
        }
        data.append(row)
    
    # Create DataFrame
    schema = StructType([
        StructField("feature_1", DoubleType(), True),
        StructField("feature_2", DoubleType(), True),
        StructField("feature_3", IntegerType(), True),
        StructField("category", StringType(), True),
        StructField("target", IntegerType(), True)
    ])
    
    df = spark_session.createDataFrame(data, schema)
    return df


class TestSparkDataProcessing:
    """Test Spark data processing capabilities."""
    
    def test_spark_dataframe_creation(self, spark_session, spark_test_data):
        """Test Spark DataFrame creation and basic operations."""
        # Verify DataFrame was created correctly
        assert spark_test_data.count() == 1000
        assert len(spark_test_data.columns) == 5
        
        # Test basic operations
        feature_stats = spark_test_data.select('feature_1').describe()
        assert feature_stats.count() == 5  # count, mean, stddev, min, max
        
        # Test filtering
        filtered_df = spark_test_data.filter(spark_test_data.target == 1)
        assert filtered_df.count() > 0
        
        # Test grouping
        category_counts = spark_test_data.groupBy('category').count()
        assert category_counts.count() == 3  # Three categories: A, B, C
    
    def test_spark_data_preprocessing(self, spark_session, spark_test_data):
        """Test data preprocessing with Spark."""
        from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler
        
        # String indexing for categorical variables
        indexer = StringIndexer(inputCol="category", outputCol="category_indexed")
        indexed_df = indexer.fit(spark_test_data).transform(spark_test_data)
        
        assert "category_indexed" in indexed_df.columns
        
        # One-hot encoding
        encoder = OneHotEncoder(inputCol="category_indexed", outputCol="category_onehot")
        encoded_df = encoder.fit(indexed_df).transform(indexed_df)
        
        assert "category_onehot" in encoded_df.columns
        
        # Feature vector assembly
        feature_cols = ['feature_1', 'feature_2', 'feature_3', 'category_onehot']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        vector_df = assembler.transform(encoded_df)
        
        assert "features" in vector_df.columns
        
        # Standard scaling
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
        scaler_model = scaler.fit(vector_df)
        scaled_df = scaler_model.transform(vector_df)
        
        assert "scaled_features" in scaled_df.columns
        assert scaled_df.count() == spark_test_data.count()
    
    def test_spark_ml_pipeline(self, spark_session, spark_test_data):
        """Test Spark ML pipeline construction."""
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
        from pyspark.ml.classification import RandomForestClassifier
        
        # Define pipeline stages
        indexer = StringIndexer(inputCol="category", outputCol="category_indexed")
        encoder = OneHotEncoder(inputCol="category_indexed", outputCol="category_onehot")
        assembler = VectorAssembler(
            inputCols=['feature_1', 'feature_2', 'feature_3', 'category_onehot'],
            outputCol="features"
        )
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol="target",
            numTrees=10,
            seed=42
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, rf])
        
        # Split data
        train_df, test_df = spark_test_data.randomSplit([0.8, 0.2], seed=42)
        
        # Fit pipeline
        model = pipeline.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Verify predictions
        assert "prediction" in predictions.columns
        assert "probability" in predictions.columns
        assert predictions.count() > 0
        
        # Evaluate model
        evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction")
        auc = evaluator.evaluate(predictions)
        
        assert 0 <= auc <= 1
        print(f"Spark ML Pipeline AUC: {auc:.4f}")


class TestSparkPipelineIntegration:
    """Test integration of Spark with the ML Pipeline Framework."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_spark_test_config(self):
        """Create Spark-specific configuration."""
        return {
            'data_source': {
                'type': 'parquet',
                'file_path': str(self.data_dir / "test_data.parquet")
            },
            'preprocessing': {
                'target_column': 'target',
                'missing_values': {
                    'strategy': 'mean'
                },
                'scaling': {
                    'method': 'standard'
                },
                'encoding': {
                    'categorical_strategy': 'onehot'
                }
            },
            'model_training': {
                'algorithm': 'RandomForestClassifier',
                'parameters': {
                    'numTrees': 20,
                    'maxDepth': 10,
                    'seed': 42
                }
            },
            'evaluation': {
                'test_size': 0.2,
                'primary_metric': 'auc'
            },
            'execution': {
                'version': 'pyspark',
                'framework': 'sparkml',
                'mode': 'train',
                'enable_mlflow': False,
                'enable_artifacts': False
            },
            'spark': {
                'app_name': 'ML Pipeline Test',
                'master': 'local[2]',
                'config': {
                    'spark.sql.adaptive.enabled': 'true',
                    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
                    'spark.driver.memory': '2g',
                    'spark.executor.memory': '1g'
                }
            }
        }
    
    def test_spark_configuration_validation(self):
        """Test Spark configuration validation."""
        config = self.create_spark_test_config()
        
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        validation_result = orchestrator.validate_config()
        
        # Should be valid if Spark is available
        assert validation_result['valid'] == True
        assert len(validation_result['errors']) == 0
    
    def test_spark_execution_plan(self):
        """Test Spark execution plan generation."""
        config = self.create_spark_test_config()
        
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        execution_plan = orchestrator.get_execution_plan()
        
        assert execution_plan['mode'] == 'train'
        assert execution_plan['framework'] == 'sparkml'
        assert execution_plan['execution_environment'] == 'pyspark'
        assert 'steps' in execution_plan
        assert len(execution_plan['steps']) > 0
        
        # Check Spark-specific resource requirements
        resources = execution_plan['resource_requirements']
        assert 'Spark cluster' in resources['environment']
    
    def test_spark_session_initialization(self, spark_session):
        """Test Spark session initialization in orchestrator."""
        config = self.create_spark_test_config()
        
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        # Test Spark session initialization
        orchestrator._initialize_spark_session()
        
        assert orchestrator.spark_session is not None
        assert orchestrator.spark_session.sparkContext.appName == 'ML Pipeline Test'
        
        # Test Spark session cleanup
        orchestrator._cleanup_execution_environment()
    
    def test_spark_data_loading_parquet(self, spark_session, spark_test_data):
        """Test Spark data loading from Parquet files."""
        # Save test data as Parquet
        parquet_path = self.data_dir / "test_data.parquet"
        
        # Convert Spark DataFrame to Pandas and save as Parquet
        pandas_df = spark_test_data.toPandas()
        pandas_df.to_parquet(parquet_path, index=False)
        
        # Test loading back into Spark
        loaded_df = spark_session.read.parquet(str(parquet_path))
        
        assert loaded_df.count() == spark_test_data.count()
        assert set(loaded_df.columns) == set(spark_test_data.columns)
    
    @pytest.mark.slow
    def test_spark_large_dataset_processing(self, spark_session):
        """Test Spark processing with larger datasets."""
        # Create larger dataset
        np.random.seed(42)
        n_samples = 50000  # Larger dataset for Spark testing
        
        # Generate data in batches to avoid memory issues
        batch_size = 10000
        all_data = []
        
        for batch in range(0, n_samples, batch_size):
            batch_data = []
            current_batch_size = min(batch_size, n_samples - batch)
            
            for i in range(current_batch_size):
                row = {
                    'feature_1': float(np.random.normal(0, 1)),
                    'feature_2': float(np.random.uniform(0, 10)),
                    'feature_3': int(np.random.randint(1, 5)),
                    'feature_4': float(np.random.exponential(2)),
                    'category_a': np.random.choice(['X', 'Y', 'Z']),
                    'category_b': np.random.choice(['P', 'Q']),
                    'target': int(np.random.choice([0, 1]))
                }
                batch_data.append(row)
            
            all_data.extend(batch_data)
        
        # Create Spark DataFrame
        schema = StructType([
            StructField("feature_1", DoubleType(), True),
            StructField("feature_2", DoubleType(), True),
            StructField("feature_3", IntegerType(), True),
            StructField("feature_4", DoubleType(), True),
            StructField("category_a", StringType(), True),
            StructField("category_b", StringType(), True),
            StructField("target", IntegerType(), True)
        ])
        
        large_df = spark_session.createDataFrame(all_data, schema)
        
        # Test basic operations on large dataset
        assert large_df.count() == n_samples
        
        # Test aggregations
        target_distribution = large_df.groupBy('target').count().collect()
        assert len(target_distribution) == 2
        
        # Test sampling
        sample_df = large_df.sample(fraction=0.1, seed=42)
        sample_count = sample_df.count()
        assert 0.05 * n_samples <= sample_count <= 0.15 * n_samples  # Allow some variance
        
        print(f"Large Spark dataset processing completed: {n_samples} rows")
        print(f"Sample size: {sample_count} rows")
    
    def test_spark_ml_feature_engineering(self, spark_session, spark_test_data):
        """Test advanced feature engineering with Spark ML."""
        from pyspark.ml.feature import (
            StringIndexer, OneHotEncoder, VectorAssembler, 
            StandardScaler, Bucketizer, QuantileDiscretizer
        )
        from pyspark.sql.functions import col, when
        
        # Feature binning
        bucketizer = Bucketizer(
            splits=[-float('inf'), -1, 0, 1, float('inf')],
            inputCol="feature_1",
            outputCol="feature_1_binned"
        )
        binned_df = bucketizer.transform(spark_test_data)
        
        assert "feature_1_binned" in binned_df.columns
        
        # Quantile discretization
        discretizer = QuantileDiscretizer(
            numBuckets=5,
            inputCol="feature_2",
            outputCol="feature_2_discretized"
        )
        discretized_df = discretizer.fit(binned_df).transform(binned_df)
        
        assert "feature_2_discretized" in discretized_df.columns
        
        # Feature interaction (manual)
        interaction_df = discretized_df.withColumn(
            "feature_interaction",
            col("feature_1") * col("feature_2")
        )
        
        assert "feature_interaction" in interaction_df.columns
        
        # Conditional features
        conditional_df = interaction_df.withColumn(
            "is_high_feature_3",
            when(col("feature_3") >= 3, 1).otherwise(0)
        )
        
        assert "is_high_feature_3" in conditional_df.columns
        assert conditional_df.count() == spark_test_data.count()
    
    def test_spark_model_evaluation(self, spark_session, spark_test_data):
        """Test model evaluation with Spark ML."""
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import StringIndexer, VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
        
        # Prepare data
        indexer = StringIndexer(inputCol="category", outputCol="category_indexed")
        assembler = VectorAssembler(
            inputCols=['feature_1', 'feature_2', 'feature_3', 'category_indexed'],
            outputCol="features"
        )
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="target",
            numTrees=20,
            seed=42
        )
        
        pipeline = Pipeline(stages=[indexer, assembler, rf])
        
        # Split data
        train_df, test_df = spark_test_data.randomSplit([0.7, 0.3], seed=42)
        
        # Train model
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        
        # Binary classification evaluation
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="target",
            rawPredictionCol="rawPrediction"
        )
        
        auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
        pr_auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})
        
        assert 0 <= auc <= 1
        assert 0 <= pr_auc <= 1
        
        # Multiclass classification evaluation
        multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol="target",
            predictionCol="prediction"
        )
        
        accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
        f1 = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "f1"})
        precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
        recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})
        
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        
        print(f"Spark ML Evaluation Results:")
        print(f"AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    def test_spark_model_persistence(self, spark_session, spark_test_data):
        """Test Spark model saving and loading."""
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import StringIndexer, VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        
        # Train model
        indexer = StringIndexer(inputCol="category", outputCol="category_indexed")
        assembler = VectorAssembler(
            inputCols=['feature_1', 'feature_2', 'feature_3', 'category_indexed'],
            outputCol="features"
        )
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="target",
            numTrees=10,
            seed=42
        )
        
        pipeline = Pipeline(stages=[indexer, assembler, rf])
        model = pipeline.fit(spark_test_data)
        
        # Save model
        model_path = self.output_dir / "spark_model"
        model.write().overwrite().save(str(model_path))
        
        # Load model
        from pyspark.ml import PipelineModel
        loaded_model = PipelineModel.load(str(model_path))
        
        # Test loaded model
        original_predictions = model.transform(spark_test_data)
        loaded_predictions = loaded_model.transform(spark_test_data)
        
        # Compare predictions (should be identical)
        original_pred_list = [row.prediction for row in original_predictions.select("prediction").collect()]
        loaded_pred_list = [row.prediction for row in loaded_predictions.select("prediction").collect()]
        
        assert original_pred_list == loaded_pred_list
        print(f"Spark model successfully saved and loaded from {model_path}")


class TestSparkPerformance:
    """Performance tests for Spark integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.slow
    @pytest.mark.timeout(600)  # 10 minute timeout
    def test_spark_large_scale_processing(self, spark_session):
        """Test Spark processing at larger scale."""
        import time
        
        # Create very large dataset
        n_samples = 100000
        n_features = 50
        
        print(f"Creating large dataset: {n_samples} samples, {n_features} features")
        
        start_time = time.time()
        
        # Generate data more efficiently using Spark's built-in functions
        from pyspark.sql.functions import rand, when, round as spark_round
        
        # Create base DataFrame with random features
        df = spark_session.range(n_samples).toDF("id")
        
        # Add random features
        for i in range(n_features):
            df = df.withColumn(f"feature_{i}", rand(seed=42+i) * 10 - 5)  # Normal-ish distribution
        
        # Add categorical features
        df = df.withColumn("category_a", when(rand(seed=100) < 0.33, "A")
                          .when(rand(seed=100) < 0.66, "B").otherwise("C"))
        df = df.withColumn("category_b", when(rand(seed=101) < 0.5, "X").otherwise("Y"))
        
        # Add target variable
        df = df.withColumn("target", (rand(seed=200) > 0.6).cast("int"))
        
        # Force computation and cache
        df.cache()
        count = df.count()
        
        data_creation_time = time.time() - start_time
        print(f"Data creation completed in {data_creation_time:.2f} seconds")
        assert count == n_samples
        
        # Test data processing operations
        start_time = time.time()
        
        # Basic statistics
        summary_stats = df.describe()
        stats_count = summary_stats.count()
        
        # Grouping operations
        target_distribution = df.groupBy("target").count()
        target_counts = target_distribution.collect()
        
        # Complex aggregations
        from pyspark.sql.functions import avg, stddev, min as spark_min, max as spark_max
        
        feature_stats = df.select([
            avg(f"feature_{i}").alias(f"avg_feature_{i}") for i in range(min(10, n_features))
        ]).collect()
        
        processing_time = time.time() - start_time
        print(f"Data processing completed in {processing_time:.2f} seconds")
        
        # Test ML operations
        start_time = time.time()
        
        from pyspark.ml.feature import StringIndexer, VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml import Pipeline
        
        # Prepare ML pipeline
        indexer_a = StringIndexer(inputCol="category_a", outputCol="category_a_indexed")
        indexer_b = StringIndexer(inputCol="category_b", outputCol="category_b_indexed")
        
        feature_cols = [f"feature_{i}" for i in range(min(20, n_features))]  # Use subset for faster training
        feature_cols.extend(["category_a_indexed", "category_b_indexed"])
        
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="target",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
        
        pipeline = Pipeline(stages=[indexer_a, indexer_b, assembler, rf])
        
        # Split data
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
        # Train model
        model = pipeline.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        prediction_count = predictions.count()
        
        ml_time = time.time() - start_time
        print(f"ML training and prediction completed in {ml_time:.2f} seconds")
        
        # Performance assertions
        total_time = data_creation_time + processing_time + ml_time
        assert total_time < 600  # Should complete within 10 minutes
        assert prediction_count > 0
        
        # Clean up cache
        df.unpersist()
        
        print(f"Large scale Spark processing summary:")
        print(f"  Dataset size: {n_samples} samples, {n_features} features")
        print(f"  Data creation: {data_creation_time:.2f}s")
        print(f"  Data processing: {processing_time:.2f}s") 
        print(f"  ML training/prediction: {ml_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
    
    def test_spark_memory_management(self, spark_session):
        """Test Spark memory management and optimization."""
        # Create DataFrame that might cause memory pressure
        n_samples = 50000
        
        # Create DataFrame with many string columns (memory intensive)
        from pyspark.sql.functions import rand, concat, lit
        
        df = spark_session.range(n_samples).toDF("id")
        
        # Add multiple string columns
        for i in range(10):
            df = df.withColumn(f"string_col_{i}", 
                              concat(lit("prefix_"), (rand(seed=i) * 1000).cast("int").cast("string")))
        
        # Add numeric columns
        for i in range(20):
            df = df.withColumn(f"numeric_col_{i}", rand(seed=100+i) * 100)
        
        # Test caching behavior
        df.cache()
        count1 = df.count()
        
        # Test repartitioning
        repartitioned_df = df.repartition(4)  # Force specific number of partitions
        count2 = repartitioned_df.count()
        
        assert count1 == count2 == n_samples
        
        # Test persistence levels
        from pyspark import StorageLevel
        
        # Test different storage levels
        df.persist(StorageLevel.MEMORY_AND_DISK)
        df.count()
        
        # Clean up
        df.unpersist()
        repartitioned_df.unpersist()
        
        print(f"Memory management test completed successfully")
        print(f"Dataset size: {n_samples} rows, {df.columns.__len__()} columns")


@pytest.mark.skipif(not pyspark, reason="PySpark not available")
class TestSparkStreamingIntegration:
    """Test Spark Streaming integration (if applicable)."""
    
    def test_spark_streaming_setup(self, spark_session):
        """Test basic Spark Streaming setup."""
        # This is a placeholder for streaming tests
        # Real streaming tests would require Kafka or other streaming sources
        
        # Test that streaming context can be created
        from pyspark.streaming import StreamingContext
        
        # Create streaming context (but don't start it in tests)
        ssc = StreamingContext(spark_session.sparkContext, 1)  # 1 second batch interval
        
        assert ssc is not None
        assert ssc.sparkContext == spark_session.sparkContext
        
        # Clean up (don't actually start streaming in tests)
        ssc.stop(stopSparkContext=False)
        
        print("Spark Streaming context creation test passed")


# Additional test markers for different test categories
pytestmark = [
    pytest.mark.integration,
    pytest.mark.spark,
    pytest.mark.skipif(not pyspark, reason="PySpark not available")
]