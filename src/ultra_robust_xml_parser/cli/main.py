"""Main CLI entry point for ultra-robust-xml command-line tool.

Provides comprehensive command-line interface for XML processing operations
including batch processing, repair, validation, and analysis.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ultra_robust_xml_parser import UltraRobustXMLParser, parse_file
from ultra_robust_xml_parser.shared.config import TokenizationConfig
from ultra_robust_xml_parser.shared.logging import get_logger


class CLIConfig:
    """Configuration management for CLI operations."""
    
    def __init__(self):
        self.parser_config = TokenizationConfig.balanced()
        self.batch_size = 50
        self.max_workers = None  # Use system default
        self.output_format = "json"
        self.verbose = False
        self.quiet = False
    
    @classmethod
    def from_file(cls, config_path: Path) -> "CLIConfig":
        """Load CLI configuration from file."""
        config = cls()
        if config_path.exists():
            try:
                with config_path.open() as f:
                    data = json.load(f)
                # Update configuration from file
                if "parser_preset" in data:
                    preset = data["parser_preset"]
                    if preset == "maximum_robustness":
                        config.parser_config = TokenizationConfig.aggressive()
                    elif preset == "performance_optimized":
                        config.parser_config = TokenizationConfig.conservative()
                    elif preset == "balanced":
                        config.parser_config = TokenizationConfig.balanced()
                
                config.batch_size = data.get("batch_size", config.batch_size)
                config.max_workers = data.get("max_workers", config.max_workers)
                config.output_format = data.get("output_format", config.output_format)
                
            except Exception as e:
                print(f"Warning: Could not load config file: {e}", file=sys.stderr)
        
        return config


class ProgressTracker:
    """Progress tracking for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.completed = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """Update progress and display if needed."""
        self.completed += increment
        current_time = time.time()
        
        # Update every second or on completion
        if current_time - self.last_update >= 1.0 or self.completed >= self.total:
            self._display_progress()
            self.last_update = current_time
    
    def _display_progress(self):
        """Display current progress."""
        if self.total == 0:
            return
        
        percentage = (self.completed / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.completed > 0 and elapsed > 0:
            rate = self.completed / elapsed
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            eta_str = f", ETA: {eta:.0f}s" if eta > 0 else ""
        else:
            eta_str = ""
        
        progress_bar = "=" * int(percentage // 2)
        progress_bar += " " * (50 - len(progress_bar))
        
        print(f"\r{self.description}: [{progress_bar}] "
              f"{percentage:.1f}% ({self.completed}/{self.total}){eta_str}",
              end="", file=sys.stderr)
        
        if self.completed >= self.total:
            print(file=sys.stderr)  # New line on completion


class XMLProcessor:
    """Core XML processing logic for CLI operations."""
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.parser = UltraRobustXMLParser(config=config.parser_config)
        self.logger = get_logger(__name__, None, "cli_processor")
    
    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single XML file and return results."""
        try:
            result = parse_file(str(file_path))
            
            return {
                "file": str(file_path),
                "success": result.success,
                "confidence": result.confidence,
                "element_count": result.element_count,
                "repair_count": result.repair_count,
                "processing_time_ms": result.performance.processing_time_ms,
                "diagnostics": [
                    {
                        "severity": diag.severity.name,
                        "message": diag.message,
                        "component": diag.component
                    } for diag in result.diagnostics
                ],
                "metadata": {
                    "encoding": getattr(result.document, "encoding", None) if result.document else None,
                    "xml_version": getattr(result.document, "version", None) if result.document else None
                }
            }
        
        except Exception as e:
            self.logger.exception("Failed to process file", extra={"file": str(file_path)})
            return {
                "file": str(file_path),
                "success": False,
                "error": str(e),
                "processing_time_ms": 0
            }
    
    def find_xml_files(self, path: Path, recursive: bool = True) -> Iterator[Path]:
        """Find XML files in path."""
        if path.is_file():
            if path.suffix.lower() in {".xml", ".xhtml", ".svg"}:
                yield path
        elif path.is_dir():
            pattern = "**/*.xml" if recursive else "*.xml"
            for xml_file in path.glob(pattern):
                if xml_file.is_file():
                    yield xml_file
            
            # Also check for other XML-like extensions
            if recursive:
                for pattern in ["**/*.xhtml", "**/*.svg"]:
                    for xml_file in path.glob(pattern):
                        if xml_file.is_file():
                            yield xml_file
    
    def batch_process(self, paths: List[Path], recursive: bool = True) -> List[Dict[str, Any]]:
        """Process multiple XML files with parallel processing."""
        # Collect all files to process
        all_files = []
        for path in paths:
            all_files.extend(self.find_xml_files(path, recursive))
        
        if not all_files:
            return []
        
        results = []
        progress = ProgressTracker(len(all_files), "Processing XML files")
        
        if len(all_files) == 1 or self.config.max_workers == 1:
            # Single-threaded processing
            for file_path in all_files:
                result = self.process_single_file(file_path)
                results.append(result)
                progress.update()
        else:
            # Multi-threaded processing
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_single_file, file_path): file_path
                    for file_path in all_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
                    progress.update()
        
        return results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="ultra-robust-xml",
        description="Ultra-robust XML parser with comprehensive analysis and repair capabilities"
    )
    
    parser.add_argument("--version", action="version", version="0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse XML files")
    parse_parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="XML files or directories to parse"
    )
    parse_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively process directories"
    )
    parse_parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "text"],
        default="json",
        help="Output format (default: json)"
    )
    parse_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (default: stdout)"
    )
    parse_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file path"
    )
    parse_parser.add_argument(
        "--preset",
        choices=["balanced", "maximum_robustness", "performance_optimized"],
        default="balanced",
        help="Parser configuration preset"
    )
    parse_parser.add_argument(
        "--workers", "-w",
        type=int,
        help="Number of parallel workers"
    )
    
    # Repair command
    repair_parser = subparsers.add_parser("repair", help="Repair malformed XML files")
    repair_parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="XML files to repair"
    )
    repair_parser.add_argument(
        "--output-dir", "-d",
        type=Path,
        help="Output directory for repaired files"
    )
    repair_parser.add_argument(
        "--suffix",
        default="_repaired",
        help="Suffix for repaired files (default: _repaired)"
    )
    repair_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file path"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate XML files")
    validate_parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="XML files to validate"
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict validation"
    )
    validate_parser.add_argument(
        "--format", "-f",
        choices=["json", "text"],
        default="text",
        help="Output format"
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output"
    )
    
    return parser


def format_results(results: List[Dict[str, Any]], format_type: str) -> str:
    """Format processing results for output."""
    if format_type == "json":
        return json.dumps(results, indent=2)
    
    elif format_type == "csv":
        if not results:
            return ""
        
        # CSV header
        lines = ["file,success,confidence,elements,repairs,time_ms,errors"]
        
        for result in results:
            error_count = len([d for d in result.get("diagnostics", []) 
                             if d.get("severity") in ["ERROR", "CRITICAL"]])
            lines.append(
                f"{result['file']},{result['success']},{result.get('confidence', 0):.3f},"
                f"{result.get('element_count', 0)},{result.get('repair_count', 0)},"
                f"{result.get('processing_time_ms', 0):.1f},{error_count}"
            )
        
        return "\n".join(lines)
    
    elif format_type == "text":
        if not results:
            return "No results to display."
        
        lines = []
        successful = sum(1 for r in results if r.get("success", False))
        
        lines.append(f"Processed {len(results)} files, {successful} successful")
        lines.append("-" * 60)
        
        for result in results:
            status = "✓" if result.get("success", False) else "✗"
            confidence = result.get("confidence", 0)
            repairs = result.get("repair_count", 0)
            time_ms = result.get("processing_time_ms", 0)
            
            lines.append(f"{status} {result['file']}")
            lines.append(f"   Confidence: {confidence:.1%}, Repairs: {repairs}, Time: {time_ms:.1f}ms")
            
            # Show errors if any
            diagnostics = result.get("diagnostics", [])
            errors = [d for d in diagnostics if d.get("severity") in ["ERROR", "CRITICAL"]]
            if errors:
                for error in errors[:3]:  # Show max 3 errors
                    lines.append(f"   Error: {error.get('message', '')}")
                if len(errors) > 3:
                    lines.append(f"   ... and {len(errors) - 3} more errors")
            
            lines.append("")
        
        return "\n".join(lines)
    
    else:
        return json.dumps(results, indent=2)


def cmd_parse(args: argparse.Namespace) -> int:
    """Handle parse command."""
    # Load configuration
    config = CLIConfig()
    if args.config and args.config.exists():
        config = CLIConfig.from_file(args.config)
    
    # Apply command-line overrides
    if args.preset:
        if args.preset == "maximum_robustness":
            config.parser_config = TokenizationConfig.aggressive()
        elif args.preset == "performance_optimized":
            config.parser_config = TokenizationConfig.conservative()
        else:
            config.parser_config = TokenizationConfig.balanced()
    
    if args.workers:
        config.max_workers = args.workers
    
    config.output_format = args.format
    
    # Process files
    processor = XMLProcessor(config)
    try:
        results = processor.batch_process(args.paths, args.recursive)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user", file=sys.stderr)
        return 1
    
    # Format and output results
    formatted_output = format_results(results, args.format)
    
    if args.output:
        try:
            args.output.write_text(formatted_output)
            print(f"Results written to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            return 1
    else:
        print(formatted_output)
    
    # Return appropriate exit code
    if not results:
        return 1
    
    successful = sum(1 for r in results if r.get("success", False))
    return 0 if successful == len(results) else 1


def cmd_repair(args: argparse.Namespace) -> int:
    """Handle repair command."""
    config = CLIConfig()
    if args.config and args.config.exists():
        config = CLIConfig.from_file(args.config)
    
    # Use aggressive configuration for repairs
    config.parser_config = TokenizationConfig.aggressive()
    
    processor = XMLProcessor(config)
    repaired_count = 0
    
    for path in args.paths:
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue
        
        try:
            result = processor.process_single_file(path)
            
            if result["success"] and result.get("repair_count", 0) > 0:
                # Create output path
                if args.output_dir:
                    output_path = args.output_dir / f"{path.stem}{args.suffix}{path.suffix}"
                else:
                    output_path = path.parent / f"{path.stem}{args.suffix}{path.suffix}"
                
                # Re-parse to get repaired XML
                parse_result = parse_file(str(path))
                if parse_result.document:
                    # Write repaired XML
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(str(parse_result.document))
                    print(f"Repaired: {path} -> {output_path}")
                    repaired_count += 1
                else:
                    print(f"Could not generate repaired output for: {path}", file=sys.stderr)
            else:
                print(f"No repairs needed for: {path}")
        
        except Exception as e:
            print(f"Failed to repair {path}: {e}", file=sys.stderr)
    
    print(f"Repaired {repaired_count} files", file=sys.stderr)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Handle validate command."""
    config = CLIConfig()
    
    # Use conservative config for strict validation
    if args.strict:
        config.parser_config = TokenizationConfig.conservative()
    else:
        config.parser_config = TokenizationConfig.balanced()
    
    processor = XMLProcessor(config)
    results = []
    
    for path in args.paths:
        if not path.exists():
            results.append({
                "file": str(path),
                "valid": False,
                "error": "File not found"
            })
            continue
        
        try:
            result = processor.process_single_file(path)
            validation_result = {
                "file": str(path),
                "valid": result["success"] and result.get("repair_count", 0) == 0,
                "confidence": result.get("confidence", 0),
                "warnings": len([d for d in result.get("diagnostics", []) 
                               if d.get("severity") == "WARNING"]),
                "errors": len([d for d in result.get("diagnostics", []) 
                             if d.get("severity") in ["ERROR", "CRITICAL"]])
            }
            
            if not validation_result["valid"]:
                # Include error details
                diagnostics = result.get("diagnostics", [])
                errors = [d.get("message", "") for d in diagnostics 
                         if d.get("severity") in ["ERROR", "CRITICAL"]]
                validation_result["error_details"] = errors[:5]  # Limit to 5 errors
            
            results.append(validation_result)
            
        except Exception as e:
            results.append({
                "file": str(path),
                "valid": False,
                "error": str(e)
            })
    
    # Output results
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        valid_count = sum(1 for r in results if r.get("valid", False))
        print(f"Validated {len(results)} files, {valid_count} valid")
        print("-" * 50)
        
        for result in results:
            status = "✓" if result.get("valid", False) else "✗"
            print(f"{status} {result['file']}")
            
            if not result.get("valid", False) and "error_details" in result:
                for error in result["error_details"][:3]:
                    print(f"   Error: {error}")
    
    # Return exit code
    valid_count = sum(1 for r in results if r.get("valid", False))
    return 0 if valid_count == len(results) else 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging verbosity
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        import logging
        logging.basicConfig(level=logging.ERROR)
    
    # Route to appropriate command handler
    try:
        if args.command == "parse":
            return cmd_parse(args)
        elif args.command == "repair":
            return cmd_repair(args)
        elif args.command == "validate":
            return cmd_validate(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user", file=sys.stderr)
        return 130  # Standard exit code for SIGINT


if __name__ == "__main__":
    sys.exit(main())