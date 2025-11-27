"""
Script to run all tests
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests(verbosity=2):
    """
    Run all unit tests.
    
    Args:
        verbosity: Test output verbosity (0, 1, or 2)
    
    Returns:
        Test results
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_specific_tests(test_pattern, verbosity=2):
    """
    Run specific tests matching a pattern.
    
    Args:
        test_pattern: Pattern to match test files (e.g., 'test_models*')
        verbosity: Test output verbosity
    
    Returns:
        Test results
    """
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern=test_pattern)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for Azerbaijani lemmatizer')
    parser.add_argument(
        '--pattern',
        type=str,
        default='test_*.py',
        help='Test file pattern (default: test_*.py)'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Test output verbosity (default: 2)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Running Azerbaijani Lemmatizer Tests")
    print("=" * 70)
    print()
    
    if args.pattern == 'test_*.py':
        result = run_all_tests(verbosity=args.verbosity)
    else:
        result = run_specific_tests(args.pattern, verbosity=args.verbosity)
    
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())