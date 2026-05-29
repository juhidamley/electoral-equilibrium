#!/usr/bin/env python3
"""
Electoral Equilibrium - Main Entry Point

This script serves as the main entry point for the Electoral Equilibrium project.
It provides a command-line interface to run various components of the pipeline.

Usage:
    python main.py --help
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from electoral.nlp.scraper import main as scraper_main
from electoral.pipeline import main as pipeline_main

def main():
    """Main entry point for the Electoral Equilibrium project."""
    parser = argparse.ArgumentParser(
        description="Electoral Equilibrium - Political Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["scraper", "pipeline"],
        default="pipeline",
        help="Which component to run"
    )
    
    parser.add_argument(
        "--config",
        default="configs/base.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (for scraper only)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "scraper":
        # Run the scraper
        scraper_main()
    else:
        # Run the pipeline
        print("Running Electoral Equilibrium Pipeline...")
        # For now, we'll just print a message since we don't have the full pipeline implemented
        print("Pipeline execution would go here")


if __name__ == "__main__":
    main()
