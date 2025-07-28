"""
Configuration template generator for ML Pipeline Framework.

This module provides template generators for different types of ML pipeline
configurations including basic, fraud detection, AutoML, and enterprise setups.
"""

from typing import Dict, Any
from datetime import datetime

class ConfigTemplateGenerator:
    """Generate configuration templates for different use cases."""
    
    def __init__(self):
        """Initialize template generator."""
        pass
    
    def generate_basic_config(self) -> Dict[str, Dict[str, Any]]:
        """Generate basic pipeline configuration template."""
        return {
            "pipeline_config": {
                "pipeline": {
                    "name": "basic-ml-pipeline",
                    "version": "1.0.0",
                    "description": "Basic machine learning pipeline",
                    "environment": "${ENVIRONMENT:dev}",
                    "log_level": "${LOG_LEVEL:INFO}"
                },
                "data_source": {
                    "type": "csv",
                    "csv": {
                        "file_paths": ["${DATA_DIR:./data}/train.csv"],
                        "separator": ",",
                        "encoding": "utf-8",
                        "header_row": 0,
                        "validate_headers": True,
                        "optimize_dtypes": True
                    }
                },
                "preprocessing": {
                    "data_quality": {
                        "enabled": True,
                        "checks": [
                            {
                                "type": "missing_values",
                                "threshold": 0.2,
                                "action": "warn"
                            },
                            {
                                "type": "duplicates",
                                "action": "remove"
                            }
                        ]
                    },
                    "cleaning": {
                        "handle_missing": {
                            "strategy": "median",
                            "columns": []
                        },
                        "handle_outliers": {
                            "method": "clip",
                            "lower_percentile": 0.01,
                            "upper_percentile": 0.99
                        }
                    },
                    "transformation": {
                        "scaling": {
                            "method": "standard",
                            "columns": []
                        },
                        "encoding": {
                            "categorical": {
                                "method": "onehot",
                                "columns": [],
                                "handle_unknown": "ignore"
                            }
                        }
                    }
                },
                "model_training": {
                    "data_split": {
                        "method": "random",
                        "train_ratio": 0.7,
                        "validation_ratio": 0.15,
                        "test_ratio": 0.15,
                        "random_state": 42
                    },
                    "target": {
                        "column": "target",
                        "type": "classification"
                    },
                    "models": [
                        {
                            "name": "random_forest",
                            "type": "sklearn_classifier",
                            "algorithm": "RandomForestClassifier",
                            "hyperparameters": {
                                "n_estimators": 100,
                                "max_depth": 10,
                                "random_state": 42
                            }
                        }
                    ],
                    "cross_validation": {
                        "enabled": True,
                        "method": "stratified_kfold",
                        "n_folds": 5,
                        "random_state": 42
                    }
                },
                "evaluation": {
                    "metrics": {
                        "classification": [
                            "accuracy", "precision", "recall", "f1_score", "roc_auc"
                        ]
                    },
                    "reports": {
                        "enabled": True,
                        "output_format": ["json", "html"],
                        "include_plots": True
                    }
                },
                "output": {
                    "model_artifacts": {
                        "save_location": "${MODEL_ARTIFACTS_DIR:./artifacts/models}",
                        "format": "joblib",
                        "versioning": True
                    },
                    "predictions": {
                        "save_location": "${PREDICTIONS_DIR:./artifacts/predictions}",
                        "format": "csv",
                        "include_probabilities": True
                    }
                }
            }
        }
    
    def generate_fraud_detection_config(self) -> Dict[str, Dict[str, Any]]:
        """Generate fraud detection specific configuration template."""
        return {
            "fraud_detection_config": {
                "pipeline": {
                    "name": "fraud-detection-pipeline",
                    "version": "1.0.0",
                    "description": "Credit card fraud detection pipeline",
                    "environment": "${ENVIRONMENT:dev}",
                    "log_level": "${LOG_LEVEL:INFO}"
                },
                "data_source": {
                    "type": "csv",
                    "csv": {
                        "file_paths": ["${DATA_DIR:./data}/credit_card_transactions.csv"],
                        "separator": ",",
                        "encoding": "utf-8",
                        "header_row": 0,
                        "date_columns": ["transaction_datetime"],
                        "date_format": "%Y-%m-%d %H:%M:%S",
                        "dtype_mapping": {
                            "customer_id": "str",
                            "merchant_id": "str",
                            "transaction_amount": "float32",
                            "is_fraud": "int8"
                        }
                    }
                },
                "preprocessing": {
                    "data_quality": {
                        "enabled": True,
                        "checks": [
                            {
                                "type": "missing_values",
                                "threshold": 0.05,
                                "action": "error"
                            },
                            {
                                "type": "duplicates",
                                "action": "remove"
                            },
                            {
                                "type": "outliers",
                                "method": "iqr",
                                "threshold": 3.0,
                                "action": "cap"
                            }
                        ]
                    },
                    "cleaning": {
                        "handle_missing": {
                            "strategy": "median",
                            "columns": ["transaction_amount", "customer_age"]
                        },
                        "handle_outliers": {
                            "method": "clip",
                            "lower_percentile": 0.001,
                            "upper_percentile": 0.999
                        }
                    }
                },
                "feature_engineering": {
                    "enabled": True,
                    "time_features": {
                        "enabled": True,
                        "datetime_column": "transaction_datetime",
                        "features": ["hour", "day_of_week", "month", "is_weekend"]
                    },
                    "derived_features": [
                        {
                            "name": "amount_to_limit_ratio",
                            "expression": "transaction_amount / credit_limit",
                            "type": "numeric"
                        },
                        {
                            "name": "velocity_score",
                            "expression": "previous_transactions_today * amount_to_limit_ratio",
                            "type": "numeric"
                        }
                    ]
                },
                "model_training": {
                    "automl_enabled": True,
                    "automl_config_path": "./configs/automl_config.yaml",
                    "data_split": {
                        "method": "time_based",
                        "train_ratio": 0.7,
                        "validation_ratio": 0.15,
                        "test_ratio": 0.15,
                        "time_column": "transaction_datetime"
                    },
                    "target": {
                        "column": "is_fraud",
                        "type": "classification",
                        "classes": [0, 1]
                    }
                },
                "evaluation": {
                    "metrics": {
                        "classification": [
                            "accuracy", "precision", "recall", "f1_score", 
                            "roc_auc", "pr_auc", "confusion_matrix"
                        ]
                    },
                    "business_metrics": {
                        "fraud_detection_rate": 0.95,
                        "false_positive_cost": 25,
                        "fraud_loss_prevented": 1000
                    }
                },
                "explainability": {
                    "enabled": True,
                    "interpretability_methods": [
                        "shap", "lime", "ale", "anchors", "counterfactuals"
                    ],
                    "shap": {
                        "enabled": True,
                        "explainer_type": "tree",
                        "sample_size": 1000
                    },
                    "lime": {
                        "enabled": True,
                        "mode": "tabular",
                        "sample_size": 100
                    }
                },
                "output": {
                    "admissible_ml_reports": True,
                    "business_metrics_focus": True,
                    "model_artifacts": {
                        "save_location": "${MODEL_ARTIFACTS_DIR:./artifacts/fraud_models}",
                        "format": "joblib",
                        "versioning": True,
                        "compression": True
                    },
                    "reports": {
                        "include_model_cards": True,
                        "include_fairness_analysis": True,
                        "include_compliance_documentation": True
                    }
                },
                "monitoring": {
                    "enabled": True,
                    "monitoring_setup": True,
                    "data_drift": {
                        "enabled": True,
                        "drift_threshold": 0.05,
                        "statistical_tests": ["ks_test", "chi2_test", "psi_test"]
                    },
                    "performance_monitoring": {
                        "enabled": True,
                        "alert_threshold": 0.1,
                        "business_impact_tracking": True
                    },
                    "fairness_monitoring": {
                        "enabled": True,
                        "protected_attributes": ["age", "gender"],
                        "fairness_metrics": ["demographic_parity", "equalized_odds"],
                        "alert_threshold": 0.8
                    }
                }
            }
        }
    
    def generate_automl_config(self) -> Dict[str, Dict[str, Any]]:
        """Generate AutoML configuration template."""
        return {
            "automl_config": {
                "automl": {
                    "name": "automl-pipeline",
                    "version": "1.0.0",
                    "description": "Automated ML pipeline with comprehensive algorithm search",
                    "random_state": 42
                },
                "general": {
                    "max_total_time": 3600,
                    "max_eval_time": 300,
                    "ensemble": True,
                    "stack_models": True,
                    "early_stopping": True,
                    "n_jobs": -1,
                    "cv_folds": 5,
                    "cv_method": "stratified_kfold",
                    "test_size": 0.2,
                    "validation_size": 0.15
                },
                "algorithm_selection": {
                    "enable_linear": True,
                    "enable_tree_based": True,
                    "enable_ensemble": True,
                    "enable_neural_networks": True,
                    "enable_naive_bayes": True,
                    "enable_svm": True,
                    "enable_neighbors": True,
                    "classification": {
                        "logistic_regression": {"enabled": True, "max_time": 300},
                        "random_forest": {"enabled": True, "max_time": 600},
                        "xgboost": {"enabled": True, "max_time": 900},
                        "lightgbm": {"enabled": True, "max_time": 900},
                        "catboost": {"enabled": True, "max_time": 900},
                        "mlp_classifier": {"enabled": True, "max_time": 1200}
                    }
                },
                "hyperparameter_spaces": {
                    "search_strategy": "bayesian",
                    "n_iterations": 100,
                    "n_initial_points": 10,
                    "acquisition_function": "ei",
                    "random_forest": {
                        "n_estimators": {"type": "int_uniform", "low": 10, "high": 500},
                        "max_depth": {"type": "int_uniform", "low": 1, "high": 20},
                        "min_samples_split": {"type": "int_uniform", "low": 2, "high": 20}
                    },
                    "xgboost": {
                        "n_estimators": {"type": "int_uniform", "low": 50, "high": 1000},
                        "max_depth": {"type": "int_uniform", "low": 3, "high": 15},
                        "learning_rate": {"type": "log_uniform", "low": 0.01, "high": 0.3}
                    }
                },
                "time_budgets": {
                    "total_budget": 3600,
                    "allocation_strategy": "dynamic",
                    "early_stopping": {
                        "enabled": True,
                        "patience": 10,
                        "min_delta": 0.001
                    }
                },
                "interpretability": {
                    "mandatory_methods": [
                        "feature_importance", "shap_values", "lime_explanations"
                    ],
                    "global_interpretability": {
                        "feature_importance": {"enabled": True, "method": "permutation"},
                        "shap_analysis": {"enabled": True, "max_samples": 1000}
                    },
                    "local_interpretability": {
                        "lime_explanations": {"enabled": True, "sample_size": 100},
                        "anchors": {"enabled": True, "threshold": 0.95}
                    }
                },
                "business_metrics": {
                    "primary_objective": "maximize_f1",
                    "cost_matrix": {
                        "binary_classification": {
                            "true_negative_value": 0,
                            "false_positive_cost": 100,
                            "false_negative_cost": 500,
                            "true_positive_value": 1000
                        }
                    },
                    "metric_weights": {
                        "accuracy": 0.2,
                        "precision": 0.3,
                        "recall": 0.3,
                        "f1_score": 0.2,
                        "business_value": 0.4
                    }
                },
                "output": {
                    "model_selection": {
                        "primary_metric": "f1_score",
                        "secondary_metrics": ["roc_auc", "precision", "recall"]
                    },
                    "results_format": {
                        "leaderboard": True,
                        "detailed_reports": True,
                        "model_comparison": True,
                        "interpretability_dashboard": True
                    },
                    "artifacts": {
                        "save_all_models": False,
                        "save_top_n_models": 5,
                        "save_ensemble_models": True,
                        "save_interpretability_data": True
                    }
                }
            }
        }
    
    def generate_enterprise_config(self) -> Dict[str, Dict[str, Any]]:
        """Generate enterprise-grade configuration template."""
        return {
            "enterprise_config": {
                "pipeline": {
                    "name": "enterprise-ml-pipeline",
                    "version": "1.0.0",
                    "description": "Enterprise-grade ML pipeline with full compliance",
                    "environment": "production",
                    "log_level": "INFO",
                    "compliance_mode": True,
                    "security_level": "high"
                },
                "data_source": {
                    "type": "postgresql",
                    "database": {
                        "connection": {
                            "host": "${DB_HOST}",
                            "port": "${DB_PORT:5432}",
                            "database": "${DB_NAME}",
                            "username": "${DB_USERNAME}",
                            "password": "${DB_PASSWORD}",
                            "schema": "${DB_SCHEMA:public}",
                            "sslmode": "require",
                            "pool_size": 10,
                            "max_overflow": 20
                        },
                        "extraction": {
                            "query": "SELECT * FROM ml_training_data WHERE created_at >= '${START_DATE}' AND created_at <= '${END_DATE}'",
                            "chunk_size": 10000,
                            "cache_data": True,
                            "encryption": True
                        }
                    }
                },
                "security": {
                    "encryption": {
                        "enabled": True,
                        "algorithm": "AES-256",
                        "key_management": "vault"
                    },
                    "access_control": {
                        "rbac_enabled": True,
                        "authentication": "jwt",
                        "authorization": "attribute_based"
                    },
                    "audit_logging": {
                        "enabled": True,
                        "level": "detailed",
                        "retention_days": 365,
                        "encryption": True
                    },
                    "data_privacy": {
                        "anonymization": True,
                        "pii_detection": True,
                        "consent_management": True
                    }
                },
                "preprocessing": {
                    "data_quality": {
                        "enabled": True,
                        "validation_level": "strict",
                        "checks": [
                            {
                                "type": "schema_validation",
                                "action": "error"
                            },
                            {
                                "type": "data_lineage",
                                "action": "log"
                            },
                            {
                                "type": "bias_detection",
                                "action": "warn"
                            }
                        ]
                    },
                    "compliance": {
                        "gdpr_compliance": True,
                        "data_retention": {
                            "enabled": True,
                            "retention_period": "7_years"
                        },
                        "right_to_be_forgotten": True
                    }
                },
                "model_training": {
                    "automl_enabled": True,
                    "automl_config_path": "./configs/enterprise_automl_config.yaml",
                    "data_split": {
                        "method": "time_based",
                        "train_ratio": 0.7,
                        "validation_ratio": 0.15,
                        "test_ratio": 0.15,
                        "stratify_on_sensitive_attributes": True
                    },
                    "fairness_constraints": {
                        "enabled": True,
                        "protected_attributes": ["age", "gender", "ethnicity"],
                        "fairness_metrics": ["demographic_parity", "equalized_odds", "calibration"],
                        "fairness_threshold": 0.8
                    },
                    "model_governance": {
                        "model_cards": True,
                        "risk_assessment": True,
                        "approval_workflow": True,
                        "version_control": True
                    }
                },
                "explainability": {
                    "enabled": True,
                    "compliance_mode": True,
                    "interpretability_methods": [
                        "shap", "lime", "ale", "anchors", "counterfactuals"
                    ],
                    "regulatory_explanations": {
                        "gdpr_explanations": True,
                        "right_to_explanation": True,
                        "automated_decision_explanations": True
                    },
                    "fairness_analysis": {
                        "enabled": True,
                        "bias_detection": True,
                        "disparate_impact_analysis": True,
                        "group_fairness_metrics": True
                    }
                },
                "output": {
                    "admissible_ml_reports": True,
                    "compliance_reports": True,
                    "model_artifacts": {
                        "save_location": "${SECURE_MODEL_STORAGE}",
                        "format": "encrypted_joblib",
                        "versioning": True,
                        "digital_signature": True,
                        "backup_enabled": True
                    },
                    "audit_trail": {
                        "enabled": True,
                        "comprehensive_logging": True,
                        "immutable_records": True
                    }
                },
                "monitoring": {
                    "enabled": True,
                    "monitoring_setup": True,
                    "comprehensive_monitoring": True,
                    "data_drift": {
                        "enabled": True,
                        "drift_threshold": 0.02,
                        "statistical_tests": ["ks_test", "chi2_test", "psi_test", "wasserstein"],
                        "alert_channels": ["email", "slack", "pagerduty"]
                    },
                    "performance_monitoring": {
                        "enabled": True,
                        "sla_monitoring": True,
                        "business_impact_tracking": True,
                        "alert_threshold": 0.05
                    },
                    "fairness_monitoring": {
                        "enabled": True,
                        "continuous_bias_monitoring": True,
                        "protected_attributes": ["age", "gender", "ethnicity"],
                        "fairness_metrics": ["demographic_parity", "equalized_odds", "calibration"],
                        "alert_threshold": 0.8
                    },
                    "security_monitoring": {
                        "enabled": True,
                        "intrusion_detection": True,
                        "anomaly_detection": True,
                        "access_monitoring": True
                    }
                },
                "deployment": {
                    "platform": "kubernetes",
                    "high_availability": True,
                    "auto_scaling": {
                        "enabled": True,
                        "min_replicas": 3,
                        "max_replicas": 20,
                        "target_cpu": 70
                    },
                    "load_balancing": True,
                    "circuit_breaker": True,
                    "health_checks": {
                        "enabled": True,
                        "readiness_probe": True,
                        "liveness_probe": True
                    }
                },
                "compliance": {
                    "frameworks": ["gdpr", "sox", "hipaa", "pci_dss"],
                    "regulatory_reporting": True,
                    "audit_readiness": True,
                    "compliance_dashboard": True,
                    "risk_management": {
                        "model_risk_rating": True,
                        "operational_risk_assessment": True,
                        "business_impact_analysis": True
                    }
                }
            }
        }
    
    def generate_config_by_type(self, config_type: str) -> Dict[str, Dict[str, Any]]:
        """Generate configuration by type.
        
        Args:
            config_type: Type of configuration to generate
            
        Returns:
            Dictionary containing configuration templates
        """
        if config_type == 'basic':
            return self.generate_basic_config()
        elif config_type == 'fraud-detection':
            return self.generate_fraud_detection_config()
        elif config_type == 'automl':
            return self.generate_automl_config()
        elif config_type == 'enterprise':
            return self.generate_enterprise_config()
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")
    
    def get_available_types(self) -> list:
        """Get list of available configuration types."""
        return ['basic', 'fraud-detection', 'automl', 'enterprise']