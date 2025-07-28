import os
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.data_context import FileDataContext
from great_expectations.expectations.expectation import ExpectationConfiguration
from great_expectations.validator.validator import Validator

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    pass


class DataValidator:
    """
    Data validation class using Great Expectations for schema validation,
    data quality checks, and documentation generation.
    """
    
    def __init__(self, context_root_dir: Optional[str] = None):
        """
        Initialize DataValidator with Great Expectations context.
        
        Args:
            context_root_dir: Path to Great Expectations context directory
        """
        self.context_root_dir = Path(context_root_dir) if context_root_dir else Path("./great_expectations")
        self.context = None
        self.datasource_name = "ml_pipeline_datasource"
        self.data_asset_name = "ml_pipeline_data_asset"
        self._initialize_context()
    
    def _initialize_context(self) -> None:
        """Initialize or create Great Expectations context."""
        try:
            if self.context_root_dir.exists():
                logger.info(f"Loading existing GX context from {self.context_root_dir}")
                self.context = gx.get_context(context_root_dir=str(self.context_root_dir))
            else:
                logger.info(f"Creating new GX context at {self.context_root_dir}")
                self.context = gx.get_context(context_root_dir=str(self.context_root_dir))
                self._setup_datasource()
        except Exception as e:
            logger.error(f"Failed to initialize GX context: {str(e)}")
            raise DataValidationError(f"Failed to initialize Great Expectations context: {str(e)}")
    
    def _setup_datasource(self) -> None:
        """Setup pandas datasource for Great Expectations."""
        try:
            datasource_config = {
                "name": self.datasource_name,
                "class_name": "Datasource",
                "execution_engine": {
                    "class_name": "PandasExecutionEngine"
                },
                "data_connectors": {
                    "runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["batch_id"]
                    }
                }
            }
            
            self.context.add_datasource(**datasource_config)
            logger.info(f"Created datasource: {self.datasource_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup datasource: {str(e)}")
            raise DataValidationError(f"Failed to setup datasource: {str(e)}")
    
    def create_expectation_suite(self, suite_name: str, overwrite: bool = False) -> None:
        """
        Create a new expectation suite.
        
        Args:
            suite_name: Name of the expectation suite
            overwrite: Whether to overwrite existing suite
        """
        try:
            if suite_name in self.context.list_expectation_suite_names() and not overwrite:
                logger.info(f"Expectation suite '{suite_name}' already exists")
                return
            
            suite = self.context.create_expectation_suite(
                expectation_suite_name=suite_name,
                overwrite_existing=overwrite
            )
            logger.info(f"Created expectation suite: {suite_name}")
            
        except Exception as e:
            logger.error(f"Failed to create expectation suite: {str(e)}")
            raise DataValidationError(f"Failed to create expectation suite: {str(e)}")
    
    def add_basic_expectations(self, suite_name: str, df: pd.DataFrame) -> None:
        """
        Add basic expectations based on DataFrame schema.
        
        Args:
            suite_name: Name of the expectation suite
            df: DataFrame to analyze for basic expectations
        """
        try:
            validator = self._get_validator(df, suite_name)
            
            # Basic table expectations
            validator.expect_table_row_count_to_be_between(
                min_value=1,
                max_value=None
            )
            
            # Column existence and type expectations
            for column in df.columns:
                validator.expect_column_to_exist(column)
                
                # Type-specific expectations
                dtype = str(df[column].dtype)
                
                if 'int' in dtype or 'float' in dtype:
                    # Numeric columns
                    validator.expect_column_values_to_be_of_type(column, dtype)
                    validator.expect_column_values_to_not_be_null(column)
                    
                    # Add min/max expectations based on data
                    min_val = df[column].min()
                    max_val = df[column].max()
                    validator.expect_column_values_to_be_between(
                        column, 
                        min_value=min_val * 0.9,  # Allow 10% variance
                        max_value=max_val * 1.1
                    )
                
                elif 'object' in dtype or 'string' in dtype:
                    # String columns
                    validator.expect_column_values_to_be_of_type(column, "object")
                    
                    # Add length expectations
                    max_length = df[column].astype(str).str.len().max()
                    if max_length > 0:
                        validator.expect_column_value_lengths_to_be_between(
                            column,
                            min_value=0,
                            max_value=max_length * 2  # Allow for growth
                        )
                
                elif 'datetime' in dtype:
                    # Datetime columns
                    validator.expect_column_values_to_be_of_type(column, "datetime64[ns]")
                    
                    # Add date range expectations
                    min_date = df[column].min()
                    max_date = df[column].max()
                    validator.expect_column_values_to_be_between(
                        column,
                        min_value=min_date,
                        max_value=max_date
                    )
            
            # Save expectations
            validator.save_expectation_suite(discard_failed_expectations=False)
            logger.info(f"Added basic expectations to suite: {suite_name}")
            
        except Exception as e:
            logger.error(f"Failed to add basic expectations: {str(e)}")
            raise DataValidationError(f"Failed to add basic expectations: {str(e)}")
    
    def add_custom_expectation(self, suite_name: str, expectation_config: Dict[str, Any]) -> None:
        """
        Add a custom expectation to the suite.
        
        Args:
            suite_name: Name of the expectation suite
            expectation_config: Expectation configuration dictionary
        """
        try:
            suite = self.context.get_expectation_suite(suite_name)
            expectation = ExpectationConfiguration(**expectation_config)
            suite.add_expectation(expectation)
            
            self.context.save_expectation_suite(suite)
            logger.info(f"Added custom expectation to suite: {suite_name}")
            
        except Exception as e:
            logger.error(f"Failed to add custom expectation: {str(e)}")
            raise DataValidationError(f"Failed to add custom expectation: {str(e)}")
    
    def add_data_quality_expectations(self, suite_name: str, df: pd.DataFrame, 
                                    quality_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Add data quality expectations based on configuration.
        
        Args:
            suite_name: Name of the expectation suite
            df: DataFrame to analyze
            quality_config: Data quality configuration
        """
        try:
            validator = self._get_validator(df, suite_name)
            
            if quality_config is None:
                quality_config = {}
            
            # Missing values threshold
            missing_threshold = quality_config.get('missing_threshold', 0.2)
            for column in df.columns:
                missing_ratio = df[column].isnull().sum() / len(df)
                if missing_ratio <= missing_threshold:
                    validator.expect_column_values_to_not_be_null(
                        column,
                        mostly=1 - missing_threshold
                    )
            
            # Duplicate rows
            if quality_config.get('check_duplicates', True):
                validator.expect_table_row_count_to_equal_other_table(
                    other_table=df.drop_duplicates()
                )
            
            # Outlier detection for numeric columns
            outlier_config = quality_config.get('outliers', {})
            outlier_method = outlier_config.get('method', 'iqr')
            outlier_threshold = outlier_config.get('threshold', 3.0)
            
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for column in numeric_columns:
                if outlier_method == 'iqr':
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    validator.expect_column_values_to_be_between(
                        column,
                        min_value=lower_bound,
                        max_value=upper_bound,
                        mostly=0.95  # Allow 5% outliers
                    )
                
                elif outlier_method == 'zscore':
                    mean_val = df[column].mean()
                    std_val = df[column].std()
                    lower_bound = mean_val - outlier_threshold * std_val
                    upper_bound = mean_val + outlier_threshold * std_val
                    
                    validator.expect_column_values_to_be_between(
                        column,
                        min_value=lower_bound,
                        max_value=upper_bound,
                        mostly=0.95
                    )
            
            # Categorical value constraints
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                unique_values = df[column].unique()
                if len(unique_values) <= 50:  # Only for low cardinality
                    validator.expect_column_values_to_be_in_set(
                        column,
                        value_set=unique_values.tolist()
                    )
            
            validator.save_expectation_suite(discard_failed_expectations=False)
            logger.info(f"Added data quality expectations to suite: {suite_name}")
            
        except Exception as e:
            logger.error(f"Failed to add data quality expectations: {str(e)}")
            raise DataValidationError(f"Failed to add data quality expectations: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame, suite_name: str, 
                     batch_id: str = "validation_batch") -> Dict[str, Any]:
        """
        Validate DataFrame against expectation suite.
        
        Args:
            df: DataFrame to validate
            suite_name: Name of the expectation suite
            batch_id: Batch identifier for this validation
            
        Returns:
            Validation results dictionary
        """
        try:
            validator = self._get_validator(df, suite_name, batch_id)
            results = validator.validate()
            
            # Extract summary statistics
            validation_summary = {
                'success': results.success,
                'statistics': results.statistics,
                'total_expectations': len(results.results),
                'successful_expectations': len([r for r in results.results if r.success]),
                'failed_expectations': len([r for r in results.results if not r.success]),
                'success_rate': len([r for r in results.results if r.success]) / len(results.results) if results.results else 0,
                'batch_id': batch_id,
                'suite_name': suite_name
            }
            
            # Add failed expectation details
            failed_expectations = []
            for result in results.results:
                if not result.success:
                    failed_expectations.append({
                        'expectation_type': result.expectation_config.expectation_type,
                        'column': result.expectation_config.kwargs.get('column'),
                        'result': result.result
                    })
            
            validation_summary['failed_expectations'] = failed_expectations
            
            logger.info(f"Validation completed. Success: {results.success}, "
                       f"Success rate: {validation_summary['success_rate']:.2%}")
            
            return validation_summary
            
        except Exception as e:
            logger.error(f"Failed to validate data: {str(e)}")
            raise DataValidationError(f"Failed to validate data: {str(e)}")
    
    def run_checkpoint(self, checkpoint_name: str, df: pd.DataFrame, 
                      suite_name: str, batch_id: str = "checkpoint_batch") -> Dict[str, Any]:
        """
        Run a checkpoint for validation and generate data docs.
        
        Args:
            checkpoint_name: Name of the checkpoint
            df: DataFrame to validate
            suite_name: Name of the expectation suite
            batch_id: Batch identifier
            
        Returns:
            Checkpoint results
        """
        try:
            # Create runtime batch request
            batch_request = RuntimeBatchRequest(
                datasource_name=self.datasource_name,
                data_connector_name="runtime_data_connector",
                data_asset_name=self.data_asset_name,
                batch_identifiers={"batch_id": batch_id},
                runtime_parameters={"batch_data": df}
            )
            
            # Create simple checkpoint
            checkpoint_config = {
                "name": checkpoint_name,
                "config_version": 1.0,
                "template_name": None,
                "run_name_template": f"{checkpoint_name}_%Y%m%d-%H%M%S",
                "expectation_suite_name": suite_name,
                "batch_request": batch_request,
                "action_list": [
                    {
                        "name": "store_validation_result",
                        "action": {"class_name": "StoreValidationResultAction"},
                    },
                    {
                        "name": "update_data_docs",
                        "action": {"class_name": "UpdateDataDocsAction"},
                    },
                ],
            }
            
            checkpoint = SimpleCheckpoint(
                f"{checkpoint_name}_{batch_id}",
                self.context,
                **checkpoint_config
            )
            
            # Run checkpoint
            results = checkpoint.run()
            
            logger.info(f"Checkpoint '{checkpoint_name}' completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run checkpoint: {str(e)}")
            raise DataValidationError(f"Failed to run checkpoint: {str(e)}")
    
    def generate_data_docs(self) -> str:
        """
        Generate data documentation.
        
        Returns:
            Path to generated data docs
        """
        try:
            self.context.build_data_docs()
            docs_path = self.context_root_dir / "uncommitted" / "data_docs" / "local_site" / "index.html"
            
            logger.info(f"Data docs generated at: {docs_path}")
            return str(docs_path)
            
        except Exception as e:
            logger.error(f"Failed to generate data docs: {str(e)}")
            raise DataValidationError(f"Failed to generate data docs: {str(e)}")
    
    def _get_validator(self, df: pd.DataFrame, suite_name: str, 
                      batch_id: str = "default_batch") -> Validator:
        """
        Get a validator for the given DataFrame and expectation suite.
        
        Args:
            df: DataFrame to validate
            suite_name: Name of the expectation suite
            batch_id: Batch identifier
            
        Returns:
            Validator object
        """
        try:
            batch_request = RuntimeBatchRequest(
                datasource_name=self.datasource_name,
                data_connector_name="runtime_data_connector",
                data_asset_name=self.data_asset_name,
                batch_identifiers={"batch_id": batch_id},
                runtime_parameters={"batch_data": df}
            )
            
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            return validator
            
        except Exception as e:
            logger.error(f"Failed to get validator: {str(e)}")
            raise DataValidationError(f"Failed to get validator: {str(e)}")
    
    def export_validation_results(self, output_path: str, 
                                format: str = "json") -> None:
        """
        Export validation results to file.
        
        Args:
            output_path: Path to save results
            format: Export format (json, html)
        """
        try:
            validation_results = self.context.get_validation_result()
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation results exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export validation results: {str(e)}")
            raise DataValidationError(f"Failed to export validation results: {str(e)}")
    
    def create_profiling_suite(self, df: pd.DataFrame, suite_name: str) -> None:
        """
        Create an expectation suite based on data profiling.
        
        Args:
            df: DataFrame to profile
            suite_name: Name of the expectation suite
        """
        try:
            from great_expectations.profile.user_configurable_profiler import UserConfigurableProfiler
            
            validator = self._get_validator(df, suite_name)
            
            profiler = UserConfigurableProfiler(
                profile_dataset=validator,
                excluded_expectations=[
                    "expect_column_values_to_be_unique"  # Often too restrictive
                ],
                ignored_columns=[],
                not_null_only=False,
                primary_or_compound_key=False,
                semantic_types_dict=None,
                table_expectations_only=False,
                value_set_threshold="MANY"
            )
            
            suite = profiler.build_suite()
            self.context.save_expectation_suite(suite)
            
            logger.info(f"Created profiling suite: {suite_name}")
            
        except Exception as e:
            logger.error(f"Failed to create profiling suite: {str(e)}")
            raise DataValidationError(f"Failed to create profiling suite: {str(e)}")