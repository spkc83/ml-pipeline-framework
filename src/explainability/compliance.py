"""Compliance and regulatory reporting for machine learning models."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
warnings.filterwarnings('ignore')


@dataclass
class ComplianceMetric:
    """Represents a compliance metric."""
    name: str
    value: float
    threshold: Optional[float]
    status: str  # 'pass', 'fail', 'warning'
    description: str
    regulation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelAuditTrail:
    """Represents model audit trail information."""
    model_id: str
    version: str
    training_date: datetime
    deployment_date: Optional[datetime]
    data_sources: List[str]
    feature_transformations: List[str]
    validation_results: Dict[str, Any]
    approval_status: str
    approver: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime objects to strings
        if isinstance(data['training_date'], datetime):
            data['training_date'] = data['training_date'].isoformat()
        if data['deployment_date'] and isinstance(data['deployment_date'], datetime):
            data['deployment_date'] = data['deployment_date'].isoformat()
        return data


class GDPRComplianceChecker:
    """Check GDPR compliance requirements."""
    
    def __init__(self):
        """Initialize GDPR compliance checker."""
        self.requirements = {
            'data_minimization': 'Use only necessary features',
            'purpose_limitation': 'Use data only for stated purpose',
            'accuracy': 'Ensure data accuracy and model performance',
            'storage_limitation': 'Implement data retention policies',
            'transparency': 'Provide explainable decisions',
            'right_to_explanation': 'Enable individual explanations'
        }
    
    def check_compliance(self, model, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str],
                        sensitive_features: Optional[List[str]] = None) -> List[ComplianceMetric]:
        """Check GDPR compliance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            sensitive_features: List of sensitive feature names
            
        Returns:
            List of compliance metrics
        """
        metrics = []
        
        # Data minimization check
        metrics.append(self._check_data_minimization(X, feature_names))
        
        # Accuracy requirement
        metrics.append(self._check_accuracy(model, X, y))
        
        # Transparency check
        metrics.append(self._check_transparency(model, feature_names))
        
        # Sensitive data check
        if sensitive_features:
            metrics.append(self._check_sensitive_data(feature_names, sensitive_features))
        
        return metrics
    
    def _check_data_minimization(self, X: np.ndarray, feature_names: List[str]) -> ComplianceMetric:
        """Check data minimization principle."""
        # Simple heuristic: flag if too many features
        n_features = X.shape[1]
        threshold = 50  # Arbitrary threshold
        
        status = 'pass' if n_features <= threshold else 'warning'
        
        return ComplianceMetric(
            name='data_minimization',
            value=n_features,
            threshold=threshold,
            status=status,
            description=f'Number of features: {n_features}. Consider feature selection.',
            regulation='GDPR Article 5(1)(c)'
        )
    
    def _check_accuracy(self, model, X: np.ndarray, y: np.ndarray) -> ComplianceMetric:
        """Check accuracy requirement."""
        try:
            predictions = model.predict(X)
            accuracy = np.mean(predictions == y)
            threshold = 0.7
            
            status = 'pass' if accuracy >= threshold else 'fail'
            
            return ComplianceMetric(
                name='accuracy',
                value=accuracy,
                threshold=threshold,
                status=status,
                description=f'Model accuracy: {accuracy:.3f}',
                regulation='GDPR Article 5(1)(d)'
            )
        except Exception as e:
            return ComplianceMetric(
                name='accuracy',
                value=0.0,
                threshold=0.7,
                status='fail',
                description=f'Could not calculate accuracy: {str(e)}',
                regulation='GDPR Article 5(1)(d)'
            )
    
    def _check_transparency(self, model, feature_names: List[str]) -> ComplianceMetric:
        """Check transparency requirement."""
        # Check if model is interpretable
        interpretable_models = ['DecisionTreeClassifier', 'LogisticRegression', 'LinearRegression']
        model_name = type(model).__name__
        
        is_interpretable = model_name in interpretable_models
        status = 'pass' if is_interpretable else 'warning'
        
        description = f'Model type: {model_name}. '
        if not is_interpretable:
            description += 'Consider using interpretable model or providing explanations.'
        
        return ComplianceMetric(
            name='transparency',
            value=1.0 if is_interpretable else 0.0,
            threshold=None,
            status=status,
            description=description,
            regulation='GDPR Article 22'
        )
    
    def _check_sensitive_data(self, feature_names: List[str], 
                             sensitive_features: List[str]) -> ComplianceMetric:
        """Check sensitive data usage."""
        sensitive_count = len([f for f in feature_names if f in sensitive_features])
        
        status = 'warning' if sensitive_count > 0 else 'pass'
        
        return ComplianceMetric(
            name='sensitive_data',
            value=sensitive_count,
            threshold=0,
            status=status,
            description=f'Using {sensitive_count} sensitive features: {[f for f in feature_names if f in sensitive_features]}',
            regulation='GDPR Article 9'
        )


class FairLendingComplianceChecker:
    """Check fair lending compliance (ECOA, Fair Credit Reporting Act)."""
    
    def __init__(self):
        """Initialize fair lending compliance checker."""
        self.protected_attributes = ['race', 'gender', 'age', 'religion', 'national_origin']
        
    def check_compliance(self, model, X: pd.DataFrame, y: np.ndarray,
                        protected_attribute: str) -> List[ComplianceMetric]:
        """Check fair lending compliance.
        
        Args:
            model: Trained model
            X: Feature matrix with column names
            y: Target labels
            protected_attribute: Name of protected attribute column
            
        Returns:
            List of compliance metrics
        """
        metrics = []
        
        if protected_attribute not in X.columns:
            return [ComplianceMetric(
                name='protected_attribute_missing',
                value=0.0,
                threshold=None,
                status='fail',
                description=f'Protected attribute {protected_attribute} not found in data',
                regulation='ECOA'
            )]
        
        # Disparate impact test
        metrics.append(self._check_disparate_impact(model, X, y, protected_attribute))
        
        # Equal opportunity test
        metrics.append(self._check_equal_opportunity(model, X, y, protected_attribute))
        
        # Feature usage check
        metrics.append(self._check_prohibited_features(X.columns.tolist()))
        
        return metrics
    
    def _check_disparate_impact(self, model, X: pd.DataFrame, y: np.ndarray,
                               protected_attribute: str) -> ComplianceMetric:
        """Check disparate impact using 80% rule."""
        try:
            predictions = model.predict(X)
            
            # Calculate approval rates by group
            groups = X[protected_attribute].unique()
            approval_rates = {}
            
            for group in groups:
                group_mask = X[protected_attribute] == group
                group_approvals = np.mean(predictions[group_mask])
                approval_rates[group] = group_approvals
            
            # Calculate disparate impact ratio
            min_rate = min(approval_rates.values())
            max_rate = max(approval_rates.values())
            
            impact_ratio = min_rate / max_rate if max_rate > 0 else 0
            threshold = 0.8  # 80% rule
            
            status = 'pass' if impact_ratio >= threshold else 'fail'
            
            return ComplianceMetric(
                name='disparate_impact',
                value=impact_ratio,
                threshold=threshold,
                status=status,
                description=f'Disparate impact ratio: {impact_ratio:.3f}. Approval rates: {approval_rates}',
                regulation='ECOA Disparate Impact'
            )
        except Exception as e:
            return ComplianceMetric(
                name='disparate_impact',
                value=0.0,
                threshold=0.8,
                status='fail',
                description=f'Error calculating disparate impact: {str(e)}',
                regulation='ECOA Disparate Impact'
            )
    
    def _check_equal_opportunity(self, model, X: pd.DataFrame, y: np.ndarray,
                                protected_attribute: str) -> ComplianceMetric:
        """Check equal opportunity (equal true positive rates)."""
        try:
            predictions = model.predict(X)
            
            groups = X[protected_attribute].unique()
            tpr_by_group = {}
            
            for group in groups:
                group_mask = X[protected_attribute] == group
                group_y_true = y[group_mask]
                group_y_pred = predictions[group_mask]
                
                # Calculate True Positive Rate
                if np.sum(group_y_true) > 0:
                    tpr = np.sum((group_y_true == 1) & (group_y_pred == 1)) / np.sum(group_y_true == 1)
                else:
                    tpr = 0.0
                
                tpr_by_group[group] = tpr
            
            # Calculate difference in TPRs
            tpr_values = list(tpr_by_group.values())
            tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
            
            threshold = 0.1  # 10% difference threshold
            status = 'pass' if tpr_diff <= threshold else 'fail'
            
            return ComplianceMetric(
                name='equal_opportunity',
                value=tpr_diff,
                threshold=threshold,
                status=status,
                description=f'TPR difference: {tpr_diff:.3f}. TPRs by group: {tpr_by_group}',
                regulation='Equal Credit Opportunity Act'
            )
        except Exception as e:
            return ComplianceMetric(
                name='equal_opportunity',
                value=1.0,
                threshold=0.1,
                status='fail',
                description=f'Error calculating equal opportunity: {str(e)}',
                regulation='Equal Credit Opportunity Act'
            )
    
    def _check_prohibited_features(self, feature_names: List[str]) -> ComplianceMetric:
        """Check for prohibited features."""
        prohibited_found = [f for f in feature_names if f.lower() in 
                           [attr.lower() for attr in self.protected_attributes]]
        
        status = 'fail' if prohibited_found else 'pass'
        
        return ComplianceMetric(
            name='prohibited_features',
            value=len(prohibited_found),
            threshold=0,
            status=status,
            description=f'Prohibited features found: {prohibited_found}' if prohibited_found else 'No prohibited features detected',
            regulation='ECOA Section 202.6'
        )


class ModelRiskManagementChecker:
    """Check model risk management requirements (SR 11-7, OCC guidelines)."""
    
    def __init__(self):
        """Initialize model risk management checker."""
        pass
    
    def check_compliance(self, model, X: np.ndarray, y: np.ndarray,
                        validation_results: Dict[str, Any]) -> List[ComplianceMetric]:
        """Check model risk management compliance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            validation_results: Model validation results
            
        Returns:
            List of compliance metrics
        """
        metrics = []
        
        # Model performance stability
        metrics.append(self._check_performance_stability(validation_results))
        
        # Model validation documentation
        metrics.append(self._check_validation_documentation(validation_results))
        
        # Model monitoring requirements
        metrics.append(self._check_monitoring_capability(model, X))
        
        # Model complexity assessment
        metrics.append(self._check_model_complexity(model, X.shape[1]))
        
        return metrics
    
    def _check_performance_stability(self, validation_results: Dict[str, Any]) -> ComplianceMetric:
        """Check model performance stability across validation sets."""
        if 'cross_validation_scores' not in validation_results:
            return ComplianceMetric(
                name='performance_stability',
                value=0.0,
                threshold=None,
                status='fail',
                description='Cross-validation results not available',
                regulation='SR 11-7'
            )
        
        cv_scores = validation_results['cross_validation_scores']
        score_std = np.std(cv_scores)
        score_mean = np.mean(cv_scores)
        
        # Coefficient of variation
        cv_coefficient = score_std / score_mean if score_mean > 0 else float('inf')
        threshold = 0.1  # 10% coefficient of variation
        
        status = 'pass' if cv_coefficient <= threshold else 'warning'
        
        return ComplianceMetric(
            name='performance_stability',
            value=cv_coefficient,
            threshold=threshold,
            status=status,
            description=f'Performance CV: {cv_coefficient:.3f}, Mean: {score_mean:.3f}, Std: {score_std:.3f}',
            regulation='SR 11-7 Model Validation'
        )
    
    def _check_validation_documentation(self, validation_results: Dict[str, Any]) -> ComplianceMetric:
        """Check validation documentation completeness."""
        required_components = [
            'train_score', 'test_score', 'cross_validation_scores', 
            'feature_importance', 'model_parameters'
        ]
        
        available_components = [comp for comp in required_components 
                               if comp in validation_results]
        
        completeness = len(available_components) / len(required_components)
        threshold = 0.8  # 80% completeness
        
        status = 'pass' if completeness >= threshold else 'fail'
        
        return ComplianceMetric(
            name='validation_documentation',
            value=completeness,
            threshold=threshold,
            status=status,
            description=f'Documentation completeness: {completeness:.1%}. Missing: {set(required_components) - set(available_components)}',
            regulation='SR 11-7 Documentation'
        )
    
    def _check_monitoring_capability(self, model, X: np.ndarray) -> ComplianceMetric:
        """Check model monitoring capability."""
        # Check if model can generate prediction probabilities (for monitoring)
        has_proba = hasattr(model, 'predict_proba')
        
        # Check if model has feature importance (for drift detection)
        has_importance = hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')
        
        monitoring_score = (int(has_proba) + int(has_importance)) / 2
        threshold = 0.5
        
        status = 'pass' if monitoring_score >= threshold else 'warning'
        
        capabilities = []
        if has_proba:
            capabilities.append('probability_output')
        if has_importance:
            capabilities.append('feature_importance')
        
        return ComplianceMetric(
            name='monitoring_capability',
            value=monitoring_score,
            threshold=threshold,
            status=status,
            description=f'Monitoring capabilities: {capabilities}',
            regulation='SR 11-7 Ongoing Monitoring'
        )
    
    def _check_model_complexity(self, model, n_features: int) -> ComplianceMetric:
        """Check model complexity for risk assessment."""
        model_name = type(model).__name__
        
        # Assign complexity scores based on model type
        complexity_scores = {
            'LinearRegression': 1,
            'LogisticRegression': 1,
            'DecisionTreeClassifier': 2,
            'RandomForestClassifier': 3,
            'GradientBoostingClassifier': 4,
            'MLPClassifier': 5,
            'SVC': 4
        }
        
        base_complexity = complexity_scores.get(model_name, 5)
        
        # Adjust for number of features
        feature_complexity = min(n_features / 100, 2)  # Cap at 2
        
        total_complexity = base_complexity + feature_complexity
        threshold = 5  # Complexity threshold for high-risk classification
        
        status = 'pass' if total_complexity <= threshold else 'warning'
        
        return ComplianceMetric(
            name='model_complexity',
            value=total_complexity,
            threshold=threshold,
            status=status,
            description=f'Model complexity: {total_complexity:.1f} (base: {base_complexity}, features: {feature_complexity:.1f})',
            regulation='SR 11-7 Model Risk Rating'
        )


class ComplianceReportGenerator:
    """Generate comprehensive compliance reports."""
    
    def __init__(self):
        """Initialize compliance report generator."""
        self.gdpr_checker = GDPRComplianceChecker()
        self.lending_checker = FairLendingComplianceChecker()
        self.risk_checker = ModelRiskManagementChecker()
        
    def generate_comprehensive_report(self, model, X: Union[np.ndarray, pd.DataFrame], 
                                    y: np.ndarray, feature_names: List[str],
                                    validation_results: Dict[str, Any],
                                    model_metadata: Optional[Dict[str, Any]] = None,
                                    compliance_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            validation_results: Model validation results
            model_metadata: Additional model metadata
            compliance_requirements: Specific compliance requirements to check
            
        Returns:
            Dictionary with comprehensive compliance report
        """
        if compliance_requirements is None:
            compliance_requirements = ['gdpr', 'fair_lending', 'model_risk']
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'data_shape': X.shape if hasattr(X, 'shape') else (len(X), len(feature_names)),
                'compliance_frameworks': compliance_requirements
            },
            'executive_summary': {},
            'detailed_findings': {},
            'recommendations': [],
            'audit_trail': None
        }
        
        all_metrics = []
        
        # GDPR Compliance
        if 'gdpr' in compliance_requirements:
            gdpr_metrics = self.gdpr_checker.check_compliance(model, X, y, feature_names)
            report['detailed_findings']['gdpr'] = [m.to_dict() for m in gdpr_metrics]
            all_metrics.extend(gdpr_metrics)
        
        # Fair Lending Compliance
        if 'fair_lending' in compliance_requirements and isinstance(X, pd.DataFrame):
            # Try to find protected attributes
            protected_attrs = ['race', 'gender', 'age', 'ethnicity']
            found_attr = None
            for attr in protected_attrs:
                if attr in X.columns:
                    found_attr = attr
                    break
            
            if found_attr:
                lending_metrics = self.lending_checker.check_compliance(model, X, y, found_attr)
                report['detailed_findings']['fair_lending'] = [m.to_dict() for m in lending_metrics]
                all_metrics.extend(lending_metrics)
            else:
                report['detailed_findings']['fair_lending'] = [{
                    'name': 'protected_attribute_missing',
                    'status': 'warning',
                    'description': 'No protected attributes found for fair lending analysis'
                }]
        
        # Model Risk Management
        if 'model_risk' in compliance_requirements:
            risk_metrics = self.risk_checker.check_compliance(model, X, y, validation_results)
            report['detailed_findings']['model_risk'] = [m.to_dict() for m in risk_metrics]
            all_metrics.extend(risk_metrics)
        
        # Executive Summary
        total_checks = len(all_metrics)
        passed_checks = len([m for m in all_metrics if m.status == 'pass'])
        failed_checks = len([m for m in all_metrics if m.status == 'fail'])
        warning_checks = len([m for m in all_metrics if m.status == 'warning'])
        
        report['executive_summary'] = {
            'overall_compliance_score': passed_checks / total_checks if total_checks > 0 else 0,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'warning_checks': warning_checks,
            'critical_issues': [m.name for m in all_metrics if m.status == 'fail'],
            'compliance_status': 'compliant' if failed_checks == 0 else 'non_compliant'
        }
        
        # Generate Recommendations
        report['recommendations'] = self._generate_recommendations(all_metrics)
        
        # Create Audit Trail
        if model_metadata:
            report['audit_trail'] = ModelAuditTrail(
                model_id=model_metadata.get('model_id', 'unknown'),
                version=model_metadata.get('version', '1.0'),
                training_date=datetime.now(),
                deployment_date=model_metadata.get('deployment_date'),
                data_sources=model_metadata.get('data_sources', ['unknown']),
                feature_transformations=model_metadata.get('transformations', []),
                validation_results=validation_results,
                approval_status='pending' if failed_checks > 0 else 'approved'
            ).to_dict()
        
        return report
    
    def _generate_recommendations(self, metrics: List[ComplianceMetric]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on compliance metrics."""
        recommendations = []
        
        for metric in metrics:
            if metric.status == 'fail':
                recommendation = {
                    'priority': 'high',
                    'metric': metric.name,
                    'issue': metric.description,
                    'regulation': metric.regulation,
                    'action': self._get_action_for_metric(metric.name)
                }
                recommendations.append(recommendation)
            elif metric.status == 'warning':
                recommendation = {
                    'priority': 'medium',
                    'metric': metric.name,
                    'issue': metric.description,
                    'regulation': metric.regulation,
                    'action': self._get_action_for_metric(metric.name)
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def _get_action_for_metric(self, metric_name: str) -> str:
        """Get recommended action for a specific metric."""
        actions = {
            'data_minimization': 'Implement feature selection to reduce the number of features used',
            'accuracy': 'Improve model performance through better data quality or model tuning',
            'transparency': 'Use interpretable models or implement explanation methods',
            'sensitive_data': 'Remove or properly handle sensitive features according to regulations',
            'disparate_impact': 'Investigate and mitigate bias in model predictions',
            'equal_opportunity': 'Ensure equal treatment across protected groups',
            'prohibited_features': 'Remove prohibited features from the model',
            'performance_stability': 'Investigate causes of performance variation and improve model robustness',
            'validation_documentation': 'Complete missing validation documentation components',
            'monitoring_capability': 'Implement proper model monitoring infrastructure',
            'model_complexity': 'Consider simplifying the model or enhancing risk management controls'
        }
        
        return actions.get(metric_name, 'Investigate and address the identified issue')
    
    def export_report(self, report: Dict[str, Any], output_path: str, 
                     format: str = 'json') -> None:
        """Export compliance report to file.
        
        Args:
            report: Compliance report dictionary
            output_path: Path to save the report
            format: Export format ('json', 'html', 'pdf')
        """
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == 'html':
            html_content = self._generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML version of compliance report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .pass {{ border-left-color: #28a745; }}
                .fail {{ border-left-color: #dc3545; }}
                .warning {{ border-left-color: #ffc107; }}
                .recommendation {{ background-color: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Compliance Report</h1>
                <p><strong>Generated:</strong> {report['report_metadata']['generation_date']}</p>
                <p><strong>Model Type:</strong> {report['report_metadata']['model_type']}</p>
                <p><strong>Compliance Status:</strong> {report['executive_summary']['compliance_status'].upper()}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Overall Compliance Score:</strong> {report['executive_summary']['overall_compliance_score']:.1%}</p>
                <p><strong>Total Checks:</strong> {report['executive_summary']['total_checks']}</p>
                <p><strong>Passed:</strong> {report['executive_summary']['passed_checks']}</p>
                <p><strong>Failed:</strong> {report['executive_summary']['failed_checks']}</p>
                <p><strong>Warnings:</strong> {report['executive_summary']['warning_checks']}</p>
            </div>
        """
        
        # Add detailed findings
        for framework, findings in report['detailed_findings'].items():
            html += f'<div class="section"><h2>{framework.replace("_", " ").title()} Compliance</h2>'
            for finding in findings:
                status_class = finding.get('status', 'unknown')
                html += f'<div class="metric {status_class}">'
                html += f'<strong>{finding.get("name", "Unknown")}</strong>: {finding.get("description", "No description")}'
                if finding.get('regulation'):
                    html += f' <em>({finding["regulation"]})</em>'
                html += '</div>'
            html += '</div>'
        
        # Add recommendations
        if report['recommendations']:
            html += '<div class="section"><h2>Recommendations</h2>'
            for rec in report['recommendations']:
                html += f'<div class="recommendation"><strong>{rec["priority"].upper()} PRIORITY:</strong> {rec["action"]}</div>'
            html += '</div>'
        
        html += '</body></html>'
        
        return html
    
    def create_compliance_dashboard(self, reports: List[Dict[str, Any]],
                                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create compliance dashboard visualization.
        
        Args:
            reports: List of compliance reports to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Compliance scores over time
        scores = [r['executive_summary']['overall_compliance_score'] for r in reports]
        dates = [r['report_metadata']['generation_date'] for r in reports]
        
        axes[0, 0].plot(range(len(scores)), scores, marker='o')
        axes[0, 0].set_title('Compliance Score Trend')
        axes[0, 0].set_ylabel('Compliance Score')
        axes[0, 0].set_xlabel('Report Number')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Status distribution
        if reports:
            latest_report = reports[-1]
            status_counts = {
                'Pass': latest_report['executive_summary']['passed_checks'],
                'Fail': latest_report['executive_summary']['failed_checks'],
                'Warning': latest_report['executive_summary']['warning_checks']
            }
            
            colors = ['green', 'red', 'orange']
            axes[0, 1].pie(status_counts.values(), labels=status_counts.keys(), 
                          colors=colors, autopct='%1.1f%%')
            axes[0, 1].set_title('Latest Compliance Status Distribution')
        
        # Framework compliance comparison
        if len(reports) > 0 and 'detailed_findings' in reports[-1]:
            frameworks = list(reports[-1]['detailed_findings'].keys())
            framework_scores = []
            
            for framework in frameworks:
                findings = reports[-1]['detailed_findings'][framework]
                if isinstance(findings, list) and len(findings) > 0:
                    passes = len([f for f in findings if f.get('status') == 'pass'])
                    total = len(findings)
                    score = passes / total if total > 0 else 0
                    framework_scores.append(score)
                else:
                    framework_scores.append(0)
            
            axes[1, 0].bar(frameworks, framework_scores, color='skyblue')
            axes[1, 0].set_title('Compliance by Framework')
            axes[1, 0].set_ylabel('Compliance Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Critical issues trend
        critical_counts = []
        for report in reports:
            critical_count = len(report['executive_summary'].get('critical_issues', []))
            critical_counts.append(critical_count)
        
        axes[1, 1].plot(range(len(critical_counts)), critical_counts, 
                       marker='s', color='red', linewidth=2)
        axes[1, 1].set_title('Critical Issues Trend')
        axes[1, 1].set_ylabel('Number of Critical Issues')
        axes[1, 1].set_xlabel('Report Number')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig