#!/usr/bin/env python3
"""
Quality checks script for the ML Pipeline Framework.

This script runs various code quality checks including linting, type checking,
and test coverage analysis with configurable thresholds.

Usage:
    python tests/quality_checks.py [options]

Examples:
    # Run all checks with default settings
    python tests/quality_checks.py

    # Run with custom coverage threshold
    python tests/quality_checks.py --coverage-threshold 85

    # Run only specific checks
    python tests/quality_checks.py --checks flake8,mypy

    # Skip specific checks
    python tests/quality_checks.py --skip-checks bandit

    # Generate detailed report
    python tests/quality_checks.py --detailed-report

    # Fix auto-fixable issues
    python tests/quality_checks.py --fix
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import tempfile
import shutil


@dataclass
class CheckResult:
    """Result of a quality check."""
    name: str
    passed: bool
    score: Optional[float] = None
    details: str = ""
    execution_time: float = 0.0
    output: str = ""
    error_output: str = ""


class QualityChecker:
    """Main quality checker class."""
    
    def __init__(self, project_root: Path, config: Dict[str, Any]):
        """
        Initialize quality checker.
        
        Args:
            project_root: Root directory of the project
            config: Configuration dictionary
        """
        self.project_root = project_root
        self.config = config
        self.src_dir = project_root / "src"
        self.tests_dir = project_root / "tests"
        self.results: List[CheckResult] = []
        
        # Ensure output directory exists
        self.output_dir = project_root / "quality_reports"
        self.output_dir.mkdir(exist_ok=True)
    
    def run_all_checks(self, checks_to_run: Optional[List[str]] = None, 
                      checks_to_skip: Optional[List[str]] = None) -> bool:
        """
        Run all quality checks.
        
        Args:
            checks_to_run: Specific checks to run (if None, run all)
            checks_to_skip: Checks to skip
            
        Returns:
            True if all checks passed, False otherwise
        """
        available_checks = [
            'flake8', 'mypy', 'bandit', 'coverage', 
            'complexity', 'imports', 'docstrings', 'security'
        ]
        
        if checks_to_run:
            checks = [c for c in checks_to_run if c in available_checks]
        else:
            checks = available_checks
        
        if checks_to_skip:
            checks = [c for c in checks if c not in checks_to_skip]
        
        print(f"Running quality checks: {', '.join(checks)}")
        print("=" * 60)
        
        # Run each check
        for check_name in checks:
            print(f"\n📋 Running {check_name}...")
            
            if check_name == 'flake8':
                result = self._run_flake8()
            elif check_name == 'mypy':
                result = self._run_mypy()
            elif check_name == 'bandit':
                result = self._run_bandit()
            elif check_name == 'coverage':
                result = self._run_coverage()
            elif check_name == 'complexity':
                result = self._run_complexity_check()
            elif check_name == 'imports':
                result = self._run_import_check()
            elif check_name == 'docstrings':
                result = self._run_docstring_check()
            elif check_name == 'security':
                result = self._run_security_check()
            else:
                result = CheckResult(
                    name=check_name,
                    passed=False,
                    details=f"Unknown check: {check_name}"
                )
            
            self.results.append(result)
            self._print_check_result(result)
        
        # Generate summary
        self._print_summary()
        
        # Generate detailed reports if requested
        if self.config.get('detailed_report', False):
            self._generate_detailed_report()
        
        # Return overall success
        return all(result.passed for result in self.results)
    
    def _run_flake8(self) -> CheckResult:
        """Run flake8 linting."""
        start_time = time.time()
        
        try:
            # Create flake8 configuration
            flake8_config = self._get_flake8_config()
            
            cmd = [
                'flake8',
                str(self.src_dir),
                '--config', flake8_config,
                '--statistics',
                '--count'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            error_count = 0
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line and line[0].isdigit():
                        error_count += int(line.split()[0])
            
            # Calculate score (100 - errors per 100 lines)
            total_lines = self._count_lines_of_code()
            if total_lines > 0:
                score = max(0, 100 - (error_count * 100 / total_lines))
            else:
                score = 100
            
            passed = error_count <= self.config.get('flake8_max_errors', 0)
            
            return CheckResult(
                name='flake8',
                passed=passed,
                score=score,
                details=f"{error_count} style issues found",
                execution_time=execution_time,
                output=result.stdout,
                error_output=result.stderr
            )
            
        except FileNotFoundError:
            return CheckResult(
                name='flake8',
                passed=False,
                details="flake8 not installed. Install with: pip install flake8",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='flake8',
                passed=False,
                details=f"Error running flake8: {e}",
                execution_time=time.time() - start_time
            )
    
    def _run_mypy(self) -> CheckResult:
        """Run mypy type checking."""
        start_time = time.time()
        
        try:
            # Create mypy configuration
            mypy_config = self._get_mypy_config()
            
            cmd = [
                'mypy',
                str(self.src_dir),
                '--config-file', mypy_config,
                '--show-error-counts'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            error_count = 0
            if "Found" in result.stdout:
                # Extract error count from "Found X errors"
                for line in result.stdout.split('\n'):
                    if "Found" in line and "error" in line:
                        try:
                            error_count = int(line.split()[1])
                        except (IndexError, ValueError):
                            pass
            
            # Calculate score
            total_files = len(list(self.src_dir.rglob("*.py")))
            if total_files > 0:
                score = max(0, 100 - (error_count * 10 / total_files))  # 10 points per error per file
            else:
                score = 100
            
            passed = error_count <= self.config.get('mypy_max_errors', 0)
            
            return CheckResult(
                name='mypy',
                passed=passed,
                score=score,
                details=f"{error_count} type errors found",
                execution_time=execution_time,
                output=result.stdout,
                error_output=result.stderr
            )
            
        except FileNotFoundError:
            return CheckResult(
                name='mypy',
                passed=False,
                details="mypy not installed. Install with: pip install mypy",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='mypy',
                passed=False,
                details=f"Error running mypy: {e}",
                execution_time=time.time() - start_time
            )
    
    def _run_bandit(self) -> CheckResult:
        """Run bandit security analysis."""
        start_time = time.time()
        
        try:
            cmd = [
                'bandit',
                '-r', str(self.src_dir),
                '-f', 'json',
                '-ll'  # Only report medium and high severity issues
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse JSON results
            try:
                if result.stdout:
                    bandit_data = json.loads(result.stdout)
                    high_issues = len([r for r in bandit_data.get('results', []) if r['issue_severity'] == 'HIGH'])
                    medium_issues = len([r for r in bandit_data.get('results', []) if r['issue_severity'] == 'MEDIUM'])
                    total_issues = high_issues + medium_issues
                else:
                    total_issues = high_issues = medium_issues = 0
            except json.JSONDecodeError:
                total_issues = high_issues = medium_issues = 0
            
            # Calculate score
            score = max(0, 100 - (high_issues * 20 + medium_issues * 10))
            
            passed = (high_issues <= self.config.get('bandit_max_high', 0) and 
                     medium_issues <= self.config.get('bandit_max_medium', 5))
            
            return CheckResult(
                name='bandit',
                passed=passed,
                score=score,
                details=f"{high_issues} high, {medium_issues} medium security issues",
                execution_time=execution_time,
                output=result.stdout,
                error_output=result.stderr
            )
            
        except FileNotFoundError:
            return CheckResult(
                name='bandit',
                passed=False,
                details="bandit not installed. Install with: pip install bandit",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='bandit',
                passed=False,
                details=f"Error running bandit: {e}",
                execution_time=time.time() - start_time
            )
    
    def _run_coverage(self) -> CheckResult:
        """Run test coverage analysis."""
        start_time = time.time()
        
        try:
            # Run tests with coverage
            cmd = [
                'python', '-m', 'pytest',
                str(self.tests_dir),
                '--cov=' + str(self.src_dir),
                '--cov-report=json',
                '--cov-report=term-missing',
                f'--cov-report=html:{self.output_dir}/coverage_html',
                '--cov-fail-under=0',  # Don't fail here, we'll check manually
                '-v'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse coverage results
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                
                # Move coverage report to output directory
                shutil.move(str(coverage_file), str(self.output_dir / 'coverage.json'))
            else:
                total_coverage = 0
            
            min_coverage = self.config.get('coverage_threshold', 80)
            passed = total_coverage >= min_coverage
            
            return CheckResult(
                name='coverage',
                passed=passed,
                score=total_coverage,
                details=f"{total_coverage:.1f}% coverage (minimum: {min_coverage}%)",
                execution_time=execution_time,
                output=result.stdout,
                error_output=result.stderr
            )
            
        except FileNotFoundError:
            return CheckResult(
                name='coverage',
                passed=False,
                details="pytest or pytest-cov not installed. Install with: pip install pytest pytest-cov",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='coverage',
                passed=False,
                details=f"Error running coverage: {e}",
                execution_time=time.time() - start_time
            )
    
    def _run_complexity_check(self) -> CheckResult:
        """Run complexity analysis."""
        start_time = time.time()
        
        try:
            cmd = [
                'radon', 'cc',
                str(self.src_dir),
                '--json'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            if result.stdout:
                complexity_data = json.loads(result.stdout)
                
                # Calculate complexity metrics
                high_complexity_functions = 0
                total_functions = 0
                max_complexity = 0
                
                for file_path, functions in complexity_data.items():
                    for func in functions:
                        total_functions += 1
                        complexity = func['complexity']
                        max_complexity = max(max_complexity, complexity)
                        
                        if complexity > self.config.get('max_complexity', 10):
                            high_complexity_functions += 1
                
                if total_functions > 0:
                    score = max(0, 100 - (high_complexity_functions * 100 / total_functions))
                else:
                    score = 100
                
                passed = high_complexity_functions <= self.config.get('max_high_complexity_functions', 5)
                
                return CheckResult(
                    name='complexity',
                    passed=passed,
                    score=score,
                    details=f"{high_complexity_functions}/{total_functions} high complexity functions (max: {max_complexity})",
                    execution_time=execution_time,
                    output=result.stdout,
                    error_output=result.stderr
                )
            else:
                return CheckResult(
                    name='complexity',
                    passed=True,
                    score=100,
                    details="No complexity issues found",
                    execution_time=execution_time
                )
            
        except FileNotFoundError:
            return CheckResult(
                name='complexity',
                passed=False,
                details="radon not installed. Install with: pip install radon",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='complexity',
                passed=False,
                details=f"Error running complexity check: {e}",
                execution_time=time.time() - start_time
            )
    
    def _run_import_check(self) -> CheckResult:
        """Check import organization and unused imports."""
        start_time = time.time()
        
        try:
            # Check with isort
            cmd = [
                'isort',
                str(self.src_dir),
                '--check-only',
                '--diff'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # isort returns 0 if imports are sorted correctly
            passed = result.returncode == 0
            
            if passed:
                details = "Import organization is correct"
                score = 100
            else:
                # Count number of files with import issues
                import_issues = result.stdout.count('--- ')
                details = f"{import_issues} files with import organization issues"
                score = max(0, 100 - (import_issues * 10))
            
            return CheckResult(
                name='imports',
                passed=passed,
                score=score,
                details=details,
                execution_time=execution_time,
                output=result.stdout,
                error_output=result.stderr
            )
            
        except FileNotFoundError:
            return CheckResult(
                name='imports',
                passed=False,
                details="isort not installed. Install with: pip install isort",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='imports',
                passed=False,
                details=f"Error checking imports: {e}",
                execution_time=time.time() - start_time
            )
    
    def _run_docstring_check(self) -> CheckResult:
        """Check docstring coverage and quality."""
        start_time = time.time()
        
        try:
            cmd = [
                'pydocstyle',
                str(self.src_dir),
                '--count'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse docstring issues
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                issue_count = 0
                for line in lines:
                    if line.startswith(str(self.src_dir)):
                        issue_count += 1
            else:
                issue_count = 0
            
            # Calculate score
            total_files = len(list(self.src_dir.rglob("*.py")))
            if total_files > 0:
                score = max(0, 100 - (issue_count * 5 / total_files))
            else:
                score = 100
            
            passed = issue_count <= self.config.get('max_docstring_issues', 50)
            
            return CheckResult(
                name='docstrings',
                passed=passed,
                score=score,
                details=f"{issue_count} docstring issues found",
                execution_time=execution_time,
                output=result.stdout,
                error_output=result.stderr
            )
            
        except FileNotFoundError:
            return CheckResult(
                name='docstrings',
                passed=False,
                details="pydocstyle not installed. Install with: pip install pydocstyle",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='docstrings',
                passed=False,
                details=f"Error checking docstrings: {e}",
                execution_time=time.time() - start_time
            )
    
    def _run_security_check(self) -> CheckResult:
        """Run additional security checks."""
        start_time = time.time()
        
        try:
            cmd = [
                'safety', 'check',
                '--json'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse safety results
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = len(safety_data)
                except json.JSONDecodeError:
                    vulnerabilities = 0
            else:
                vulnerabilities = 0
            
            passed = vulnerabilities == 0
            score = 100 if passed else max(0, 100 - vulnerabilities * 20)
            
            return CheckResult(
                name='security',
                passed=passed,
                score=score,
                details=f"{vulnerabilities} known security vulnerabilities in dependencies",
                execution_time=execution_time,
                output=result.stdout,
                error_output=result.stderr
            )
            
        except FileNotFoundError:
            return CheckResult(
                name='security',
                passed=False,
                details="safety not installed. Install with: pip install safety",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CheckResult(
                name='security',
                passed=False,
                details=f"Error running security check: {e}",
                execution_time=time.time() - start_time
            )
    
    def _get_flake8_config(self) -> str:
        """Create flake8 configuration file."""
        config_content = """
[flake8]
max-line-length = 100
max-complexity = 10
ignore = 
    E203,  # whitespace before ':'
    W503,  # line break before binary operator
    F401,  # imported but unused (handled by isort)
exclude = 
    .git,
    __pycache__,
    .pytest_cache,
    .mypy_cache,
    build,
    dist,
    *.egg-info
per-file-ignores =
    __init__.py:F401
    tests/*:S101,S311  # Allow assert and random in tests
"""
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
        config_file.write(config_content)
        config_file.close()
        
        return config_file.name
    
    def _get_mypy_config(self) -> str:
        """Create mypy configuration file."""
        config_content = """
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False
check_untyped_defs = True
disallow_untyped_decorators = False
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
ignore_missing_imports = True

[mypy-tests.*]
ignore_errors = True
"""
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
        config_file.write(config_content)
        config_file.close()
        
        return config_file.name
    
    def _count_lines_of_code(self) -> int:
        """Count total lines of code in the project."""
        total_lines = 0
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except Exception:
                pass
        return total_lines
    
    def _print_check_result(self, result: CheckResult):
        """Print the result of a quality check."""
        status_icon = "✅" if result.passed else "❌"
        score_text = f" ({result.score:.1f}/100)" if result.score is not None else ""
        time_text = f" [{result.execution_time:.2f}s]"
        
        print(f"{status_icon} {result.name}{score_text}{time_text}: {result.details}")
        
        if not result.passed and result.error_output:
            print(f"   Error: {result.error_output[:200]}...")
    
    def _print_summary(self):
        """Print summary of all quality checks."""
        print("\n" + "=" * 60)
        print("📊 QUALITY CHECK SUMMARY")
        print("=" * 60)
        
        passed_checks = [r for r in self.results if r.passed]
        failed_checks = [r for r in self.results if not r.passed]
        
        print(f"✅ Passed: {len(passed_checks)}")
        print(f"❌ Failed: {len(failed_checks)}")
        print(f"⏱️  Total time: {sum(r.execution_time for r in self.results):.2f}s")
        
        # Calculate overall score
        scores = [r.score for r in self.results if r.score is not None]
        if scores:
            overall_score = sum(scores) / len(scores)
            print(f"📈 Overall score: {overall_score:.1f}/100")
        
        # Print failed checks
        if failed_checks:
            print(f"\n❌ Failed checks:")
            for result in failed_checks:
                print(f"   • {result.name}: {result.details}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if any(r.name == 'coverage' and not r.passed for r in self.results):
            print("   • Increase test coverage by adding more unit tests")
        
        if any(r.name == 'flake8' and not r.passed for r in self.results):
            print("   • Fix code style issues with: autopep8 --in-place --recursive src/")
        
        if any(r.name == 'imports' and not r.passed for r in self.results):
            print("   • Fix import organization with: isort src/")
        
        if any(r.name == 'mypy' and not r.passed for r in self.results):
            print("   • Add type hints to improve code clarity and catch type errors")
        
        print()
    
    def _generate_detailed_report(self):
        """Generate detailed quality report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"quality_report_{timestamp}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'config': self.config,
            'summary': {
                'total_checks': len(self.results),
                'passed_checks': len([r for r in self.results if r.passed]),
                'failed_checks': len([r for r in self.results if not r.passed]),
                'overall_score': sum(r.score for r in self.results if r.score is not None) / 
                               len([r for r in self.results if r.score is not None]) if self.results else 0,
                'total_execution_time': sum(r.execution_time for r in self.results)
            },
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'score': r.score,
                    'details': r.details,
                    'execution_time': r.execution_time,
                    'output': r.output,
                    'error_output': r.error_output
                }
                for r in self.results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    def fix_auto_fixable_issues(self):
        """Automatically fix issues that can be auto-fixed."""
        print("🔧 Attempting to auto-fix issues...")
        
        # Fix import organization
        try:
            subprocess.run(['isort', str(self.src_dir)], check=True)
            print("✅ Fixed import organization with isort")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Could not fix imports (isort not available or failed)")
        
        # Fix code formatting
        try:
            subprocess.run(['autopep8', '--in-place', '--recursive', str(self.src_dir)], check=True)
            print("✅ Fixed code formatting with autopep8")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Could not fix formatting (autopep8 not available or failed)")
        
        # Fix trailing whitespace and line endings
        try:
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Remove trailing whitespace and ensure single newline at end
                lines = [line.rstrip() for line in content.splitlines()]
                fixed_content = '\n'.join(lines) + '\n'
                
                if content != fixed_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
            
            print("✅ Fixed trailing whitespace and line endings")
        except Exception as e:
            print(f"❌ Could not fix whitespace issues: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run quality checks for the ML Pipeline Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--coverage-threshold',
        type=float,
        default=80,
        help='Minimum test coverage percentage (default: 80)'
    )
    
    parser.add_argument(
        '--flake8-max-errors',
        type=int,
        default=0,
        help='Maximum allowed flake8 errors (default: 0)'
    )
    
    parser.add_argument(
        '--mypy-max-errors',
        type=int,
        default=0,
        help='Maximum allowed mypy errors (default: 0)'
    )
    
    parser.add_argument(
        '--max-complexity',
        type=int,
        default=10,
        help='Maximum allowed cyclomatic complexity (default: 10)'
    )
    
    parser.add_argument(
        '--checks',
        type=str,
        help='Comma-separated list of checks to run (flake8,mypy,bandit,coverage,complexity,imports,docstrings,security)'
    )
    
    parser.add_argument(
        '--skip-checks',
        type=str,
        help='Comma-separated list of checks to skip'
    )
    
    parser.add_argument(
        '--detailed-report',
        action='store_true',
        help='Generate detailed JSON report'
    )
    
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to automatically fix issues'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        help='Project root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        # Try to find project root by looking for key files
        current = Path.cwd()
        while current != current.parent:
            if (current / 'src').exists() and (current / 'tests').exists():
                project_root = current
                break
            current = current.parent
        else:
            project_root = Path.cwd()
    
    # Build configuration
    config = {
        'coverage_threshold': args.coverage_threshold,
        'flake8_max_errors': args.flake8_max_errors,
        'mypy_max_errors': args.mypy_max_errors,
        'max_complexity': args.max_complexity,
        'detailed_report': args.detailed_report,
        'bandit_max_high': 0,
        'bandit_max_medium': 5,
        'max_high_complexity_functions': 5,
        'max_docstring_issues': 50
    }
    
    # Parse check lists
    checks_to_run = None
    if args.checks:
        checks_to_run = [c.strip() for c in args.checks.split(',')]
    
    checks_to_skip = None
    if args.skip_checks:
        checks_to_skip = [c.strip() for c in args.skip_checks.split(',')]
    
    print(f"🔍 ML Pipeline Framework Quality Checker")
    print(f"📁 Project root: {project_root}")
    print(f"📊 Coverage threshold: {args.coverage_threshold}%")
    print()
    
    # Initialize checker
    checker = QualityChecker(project_root, config)
    
    # Auto-fix issues if requested
    if args.fix:
        checker.fix_auto_fixable_issues()
        print()
    
    # Run quality checks
    all_passed = checker.run_all_checks(checks_to_run, checks_to_skip)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()