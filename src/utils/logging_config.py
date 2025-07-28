"""
Logging configuration module for the ML Pipeline Framework.

This module provides structured logging with different levels for components,
MLflow integration, and configurable output formats.
"""

import logging
import logging.config
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import time

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentType(Enum):
    """ML pipeline component types for targeted logging."""
    DATA_ACCESS = "data_access"
    PREPROCESSING = "preprocessing"
    MODEL = "model"
    EVALUATION = "evaluation"
    EXPLAINABILITY = "explainability"
    ORCHESTRATOR = "orchestrator"
    UTILS = "utils"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    component: str
    message: str
    module: str
    function: str
    line_number: int
    process_id: int
    thread_id: int
    user: Optional[str] = None
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    stage: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    extra_data: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


class MLflowLogHandler(logging.Handler):
    """Custom handler to send logs to MLflow."""
    
    def __init__(self, min_level: str = "WARNING"):
        super().__init__()
        self.min_level = getattr(logging, min_level.upper())
        self.buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        self.last_flush = time.time()
        
    def emit(self, record):
        """Emit a log record to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
            
        if record.levelno < self.min_level:
            return
            
        try:
            # Create structured log entry
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': record.levelname,
                'component': getattr(record, 'component', 'unknown'),
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line_number': record.lineno,
                'logger_name': record.name
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = ''.join(traceback.format_exception(*record.exc_info))
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'message', 'exc_info', 'exc_text', 
                              'stack_info']:
                    log_entry[f'extra_{key}'] = str(value)
            
            # Buffer the log entry
            self.buffer.append(log_entry)
            
            # Check if we need to flush
            current_time = time.time()
            if (len(self.buffer) >= self.buffer_size or 
                current_time - self.last_flush >= self.flush_interval or
                record.levelno >= logging.ERROR):
                self.flush_to_mlflow()
                
        except Exception:
            # Don't let logging errors break the application
            pass
    
    def flush_to_mlflow(self):
        """Flush buffered logs to MLflow."""
        if not self.buffer or not MLFLOW_AVAILABLE:
            return
            
        try:
            # Get active run
            active_run = mlflow.active_run()
            if active_run:
                # Log each entry as a metric or artifact
                for i, entry in enumerate(self.buffer):
                    # Log critical/error messages as metrics
                    if entry['level'] in ['ERROR', 'CRITICAL']:
                        mlflow.log_metric(f"error_count_{entry['component']}", 1, step=int(time.time()))
                    
                    # Log as artifact for detailed analysis
                    if entry['level'] in ['ERROR', 'CRITICAL', 'WARNING']:
                        artifact_name = f"logs/{entry['level'].lower()}_{int(time.time())}_{i}.json"
                        temp_file = Path.cwd() / "temp_log.json"
                        
                        with open(temp_file, 'w') as f:
                            json.dump(entry, f, indent=2)
                        
                        mlflow.log_artifact(str(temp_file), artifact_name)
                        temp_file.unlink()  # Clean up
            
            # Clear buffer
            self.buffer.clear()
            self.last_flush = time.time()
            
        except Exception:
            # Don't let logging errors break the application
            pass


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record):
        """Format the log record as structured JSON."""
        # Create base log entry
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            level=record.levelname,
            component=getattr(record, 'component', 'unknown'),
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            process_id=record.process,
            thread_id=record.thread,
            user=getattr(record, 'user', None),
            session_id=getattr(record, 'session_id', None),
            experiment_id=getattr(record, 'experiment_id', None),
            run_id=getattr(record, 'run_id', None),
            stage=getattr(record, 'stage', None),
            execution_time=getattr(record, 'execution_time', None),
            memory_usage=getattr(record, 'memory_usage', None)
        )
        
        # Add exception info
        if record.exc_info:
            log_entry.stack_trace = ''.join(traceback.format_exception(*record.exc_info))
        
        # Add extra data
        if self.include_extra:
            extra_data = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'message', 'exc_info', 'exc_text', 
                              'stack_info', 'component', 'user', 'session_id', 
                              'experiment_id', 'run_id', 'stage', 'execution_time', 
                              'memory_usage']:
                    extra_data[key] = value
            
            if extra_data:
                log_entry.extra_data = extra_data
        
        # Convert to JSON
        return json.dumps(asdict(log_entry), default=str, ensure_ascii=False)


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically adds component information."""
    
    def __init__(self, logger, component: ComponentType, extra: Optional[Dict[str, Any]] = None):
        self.component = component
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Process the log message and add component information."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra']['component'] = self.component.value
        
        # Add any adapter extra data
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs
    
    def log_performance(self, level: int, message: str, execution_time: float, 
                       memory_usage: Optional[float] = None, **kwargs):
        """Log performance metrics."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra']['execution_time'] = execution_time
        if memory_usage is not None:
            kwargs['extra']['memory_usage'] = memory_usage
        
        self.log(level, message, **kwargs)
    
    def log_experiment(self, level: int, message: str, experiment_id: Optional[str] = None,
                      run_id: Optional[str] = None, **kwargs):
        """Log with experiment context."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        if experiment_id:
            kwargs['extra']['experiment_id'] = experiment_id
        if run_id:
            kwargs['extra']['run_id'] = run_id
        
        # Try to get from MLflow if not provided
        if MLFLOW_AVAILABLE and (not experiment_id or not run_id):
            try:
                active_run = mlflow.active_run()
                if active_run:
                    if not experiment_id:
                        kwargs['extra']['experiment_id'] = active_run.info.experiment_id
                    if not run_id:
                        kwargs['extra']['run_id'] = active_run.info.run_id
            except Exception:
                pass
        
        self.log(level, message, **kwargs)


class LoggingConfig:
    """Main logging configuration class."""
    
    def __init__(self, 
                 log_level: Union[str, LogLevel] = LogLevel.INFO,
                 log_dir: Optional[Union[str, Path]] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True,
                 enable_mlflow: bool = True,
                 file_rotation: bool = True,
                 max_file_size: str = "10MB",
                 backup_count: int = 5,
                 component_levels: Optional[Dict[ComponentType, Union[str, LogLevel]]] = None):
        """
        Initialize logging configuration.
        
        Args:
            log_level: Default log level
            log_dir: Directory for log files
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_json: Enable JSON structured logging
            enable_mlflow: Enable MLflow integration
            file_rotation: Enable log file rotation
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            component_levels: Specific log levels for components
        """
        self.log_level = log_level if isinstance(log_level, LogLevel) else LogLevel(log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.file_rotation = file_rotation
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.component_levels = component_levels or {}
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Store configured loggers
        self._loggers = {}
        
        # Setup root configuration
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup the root logger configuration."""
        # Create logging configuration
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'json': {
                    '()': StructuredFormatter,
                    'include_extra': True
                }
            },
            'handlers': {},
            'loggers': {},
            'root': {
                'level': self.log_level.value,
                'handlers': []
            }
        }
        
        # Console handler
        if self.enable_console:
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'level': self.log_level.value,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
            config['root']['handlers'].append('console')
        
        # File handlers
        if self.enable_file:
            # Standard text log
            if self.file_rotation:
                config['handlers']['file'] = {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.log_level.value,
                    'formatter': 'detailed',
                    'filename': str(self.log_dir / 'pipeline.log'),
                    'maxBytes': self._parse_file_size(self.max_file_size),
                    'backupCount': self.backup_count
                }
            else:
                config['handlers']['file'] = {
                    'class': 'logging.FileHandler',
                    'level': self.log_level.value,
                    'formatter': 'detailed',
                    'filename': str(self.log_dir / 'pipeline.log')
                }
            config['root']['handlers'].append('file')
            
            # JSON structured log
            if self.enable_json:
                if self.file_rotation:
                    config['handlers']['json_file'] = {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': self.log_level.value,
                        'formatter': 'json',
                        'filename': str(self.log_dir / 'pipeline.json'),
                        'maxBytes': self._parse_file_size(self.max_file_size),
                        'backupCount': self.backup_count
                    }
                else:
                    config['handlers']['json_file'] = {
                        'class': 'logging.FileHandler',
                        'level': self.log_level.value,
                        'formatter': 'json',
                        'filename': str(self.log_dir / 'pipeline.json')
                    }
                config['root']['handlers'].append('json_file')
        
        # MLflow handler
        if self.enable_mlflow:
            config['handlers']['mlflow'] = {
                '()': MLflowLogHandler,
                'min_level': 'WARNING'
            }
            config['root']['handlers'].append('mlflow')
        
        # Component-specific loggers
        for component, level in self.component_levels.items():
            level_value = level.value if isinstance(level, LogLevel) else level.upper()
            config['loggers'][f'ml_pipeline.{component.value}'] = {
                'level': level_value,
                'handlers': config['root']['handlers'],
                'propagate': False
            }
        
        # Apply configuration
        logging.config.dictConfig(config)
        
        # Log configuration success
        logger = logging.getLogger(__name__)
        logger.info("Logging configuration initialized")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info(f"Default log level: {self.log_level.value}")
        
        if self.enable_mlflow:
            logger.info("MLflow logging integration enabled")
        
        if self.component_levels:
            logger.info(f"Component-specific levels: {self.component_levels}")
    
    def get_logger(self, component: ComponentType, name: Optional[str] = None) -> ComponentLoggerAdapter:
        """
        Get a logger for a specific component.
        
        Args:
            component: Component type
            name: Optional specific logger name
            
        Returns:
            ComponentLoggerAdapter configured for the component
        """
        logger_name = f'ml_pipeline.{component.value}'
        if name:
            logger_name += f'.{name}'
        
        if logger_name not in self._loggers:
            base_logger = logging.getLogger(logger_name)
            adapter = ComponentLoggerAdapter(base_logger, component)
            self._loggers[logger_name] = adapter
        
        return self._loggers[logger_name]
    
    def get_performance_logger(self, component: ComponentType) -> 'PerformanceLogger':
        """Get a performance logger for detailed timing and resource tracking."""
        return PerformanceLogger(self.get_logger(component))
    
    def _parse_file_size(self, size_str: str) -> int:
        """Parse file size string to bytes."""
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def update_component_level(self, component: ComponentType, level: Union[str, LogLevel]):
        """Update log level for a specific component."""
        level_value = level.value if isinstance(level, LogLevel) else level.upper()
        
        logger_name = f'ml_pipeline.{component.value}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(level_value)
        
        self.component_levels[component] = level
        
        # Log the change
        root_logger = logging.getLogger(__name__)
        root_logger.info(f"Updated log level for {component.value} to {level_value}")
    
    def create_experiment_context(self, experiment_name: str, 
                                 tags: Optional[Dict[str, str]] = None) -> 'ExperimentContext':
        """Create an experiment context for structured logging."""
        return ExperimentContext(self, experiment_name, tags)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'configuration': {
                'log_level': self.log_level.value,
                'log_dir': str(self.log_dir),
                'handlers_enabled': {
                    'console': self.enable_console,
                    'file': self.enable_file,
                    'json': self.enable_json,
                    'mlflow': self.enable_mlflow
                },
                'component_levels': {
                    comp.value: level.value if isinstance(level, LogLevel) else level
                    for comp, level in self.component_levels.items()
                }
            },
            'log_files': []
        }
        
        # Get log file information
        if self.log_dir.exists():
            for log_file in self.log_dir.glob('*.log*'):
                try:
                    file_stats = log_file.stat()
                    stats['log_files'].append({
                        'name': log_file.name,
                        'size_bytes': file_stats.st_size,
                        'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    })
                except Exception:
                    pass
        
        return stats


class PerformanceLogger:
    """Logger for detailed performance tracking."""
    
    def __init__(self, logger: ComponentLoggerAdapter):
        self.logger = logger
        self._start_times = {}
        self._memory_tracker = MemoryTracker() if self._psutil_available() else None
    
    def _psutil_available(self) -> bool:
        """Check if psutil is available for memory tracking."""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{int(time.time() * 1000000)}"
        self._start_times[timer_id] = time.time()
        
        if self._memory_tracker:
            self._memory_tracker.start_tracking(timer_id)
        
        self.logger.debug(f"Started timing operation: {operation}", extra={'timer_id': timer_id})
        return timer_id
    
    def end_timer(self, timer_id: str, operation: str, success: bool = True):
        """End timing an operation and log results."""
        if timer_id not in self._start_times:
            self.logger.warning(f"Timer {timer_id} not found for operation {operation}")
            return
        
        execution_time = time.time() - self._start_times[timer_id]
        del self._start_times[timer_id]
        
        memory_usage = None
        if self._memory_tracker:
            memory_usage = self._memory_tracker.end_tracking(timer_id)
        
        level = logging.INFO if success else logging.WARNING
        status = "completed" if success else "failed"
        
        self.logger.log_performance(
            level,
            f"Operation {operation} {status}",
            execution_time,
            memory_usage,
            timer_id=timer_id,
            operation=operation,
            success=success
        )
    
    def log_metric(self, metric_name: str, value: float, unit: Optional[str] = None):
        """Log a performance metric."""
        extra = {'metric_name': metric_name, 'metric_value': value}
        if unit:
            extra['metric_unit'] = unit
        
        self.logger.info(f"Metric {metric_name}: {value} {unit or ''}", extra=extra)


class MemoryTracker:
    """Track memory usage for operations."""
    
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self._tracking = {}
        except ImportError:
            self.psutil = None
    
    def start_tracking(self, tracking_id: str):
        """Start tracking memory for an operation."""
        if not self.psutil:
            return
        
        try:
            process = self.psutil.Process()
            memory_info = process.memory_info()
            self._tracking[tracking_id] = {
                'start_rss': memory_info.rss,
                'start_vms': memory_info.vms,
                'start_time': time.time()
            }
        except Exception:
            pass
    
    def end_tracking(self, tracking_id: str) -> Optional[float]:
        """End tracking and return peak memory usage in MB."""
        if not self.psutil or tracking_id not in self._tracking:
            return None
        
        try:
            process = self.psutil.Process()
            memory_info = process.memory_info()
            
            start_info = self._tracking[tracking_id]
            memory_delta = (memory_info.rss - start_info['start_rss']) / (1024 * 1024)
            
            del self._tracking[tracking_id]
            return memory_delta
            
        except Exception:
            return None


class ExperimentContext:
    """Context manager for experiment-aware logging."""
    
    def __init__(self, logging_config: LoggingConfig, experiment_name: str, 
                 tags: Optional[Dict[str, str]] = None):
        self.logging_config = logging_config
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.experiment_id = None
        self.run_id = None
        self._original_context = {}
    
    def __enter__(self):
        """Enter experiment context."""
        if MLFLOW_AVAILABLE:
            try:
                # Set or create experiment
                mlflow.set_experiment(self.experiment_name)
                
                # Start run
                mlflow.start_run(tags=self.tags)
                
                # Get experiment and run info
                active_run = mlflow.active_run()
                if active_run:
                    self.experiment_id = active_run.info.experiment_id
                    self.run_id = active_run.info.run_id
                    
                    # Log experiment start
                    logger = self.logging_config.get_logger(ComponentType.SYSTEM)
                    logger.log_experiment(
                        logging.INFO,
                        f"Started experiment: {self.experiment_name}",
                        self.experiment_id,
                        self.run_id
                    )
            
            except Exception as e:
                logger = self.logging_config.get_logger(ComponentType.SYSTEM)
                logger.error(f"Failed to start MLflow experiment: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit experiment context."""
        if MLFLOW_AVAILABLE:
            try:
                # Log experiment end
                logger = self.logging_config.get_logger(ComponentType.SYSTEM)
                
                if exc_type is None:
                    logger.log_experiment(
                        logging.INFO,
                        f"Completed experiment: {self.experiment_name}",
                        self.experiment_id,
                        self.run_id
                    )
                else:
                    logger.log_experiment(
                        logging.ERROR,
                        f"Experiment failed: {self.experiment_name} - {exc_val}",
                        self.experiment_id,
                        self.run_id,
                        exception_type=str(exc_type),
                        exception_message=str(exc_val)
                    )
                
                # End MLflow run
                mlflow.end_run()
                
            except Exception as e:
                logger = self.logging_config.get_logger(ComponentType.SYSTEM)
                logger.error(f"Failed to end MLflow experiment: {e}")


# Global logging configuration instance
_global_config: Optional[LoggingConfig] = None


def initialize_logging(config: Optional[LoggingConfig] = None, **kwargs) -> LoggingConfig:
    """
    Initialize global logging configuration.
    
    Args:
        config: Logging configuration instance
        **kwargs: Configuration parameters if config is None
        
    Returns:
        LoggingConfig instance
    """
    global _global_config
    
    if config is None:
        config = LoggingConfig(**kwargs)
    
    _global_config = config
    return config


def get_logger(component: ComponentType, name: Optional[str] = None) -> ComponentLoggerAdapter:
    """
    Get a logger for a component using global configuration.
    
    Args:
        component: Component type
        name: Optional specific logger name
        
    Returns:
        ComponentLoggerAdapter
    """
    if _global_config is None:
        initialize_logging()
    
    return _global_config.get_logger(component, name)


def get_performance_logger(component: ComponentType) -> PerformanceLogger:
    """Get a performance logger using global configuration."""
    if _global_config is None:
        initialize_logging()
    
    return _global_config.get_performance_logger(component)


# Convenience functions for common components
def get_data_logger(name: Optional[str] = None) -> ComponentLoggerAdapter:
    """Get logger for data access components."""
    return get_logger(ComponentType.DATA_ACCESS, name)


def get_model_logger(name: Optional[str] = None) -> ComponentLoggerAdapter:
    """Get logger for model components."""
    return get_logger(ComponentType.MODEL, name)


def get_preprocessing_logger(name: Optional[str] = None) -> ComponentLoggerAdapter:
    """Get logger for preprocessing components."""
    return get_logger(ComponentType.PREPROCESSING, name)


def get_evaluation_logger(name: Optional[str] = None) -> ComponentLoggerAdapter:
    """Get logger for evaluation components."""
    return get_logger(ComponentType.EVALUATION, name)


def get_system_logger(name: Optional[str] = None) -> ComponentLoggerAdapter:
    """Get logger for system components."""
    return get_logger(ComponentType.SYSTEM, name)