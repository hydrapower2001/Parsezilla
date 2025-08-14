"""Tests for the CLI main module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ultra_robust_xml_parser.cli.main import (
    CLIConfig,
    ProgressTracker,
    XMLProcessor,
    create_argument_parser,
    format_results,
    main,
)


class TestCLIConfig:
    """Test CLI configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CLIConfig()
        assert config.batch_size == 50
        assert config.max_workers is None
        assert config.output_format == "json"
        assert config.verbose is False
        assert config.quiet is False
    
    def test_config_from_file(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "parser_preset": "maximum_robustness",
                "batch_size": 100,
                "max_workers": 4,
                "output_format": "csv"
            }
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = CLIConfig.from_file(config_path)
            assert config.batch_size == 100
            assert config.max_workers == 4
            assert config.output_format == "csv"
        finally:
            config_path.unlink()
    
    def test_config_from_nonexistent_file(self):
        """Test handling non-existent config file."""
        config = CLIConfig.from_file(Path("nonexistent.json"))
        # Should use default values
        assert config.batch_size == 50
        assert config.output_format == "json"


class TestProgressTracker:
    """Test progress tracking functionality."""
    
    def test_progress_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(100, "Test")
        assert tracker.total == 100
        assert tracker.completed == 0
        assert tracker.description == "Test"
    
    def test_progress_update(self):
        """Test progress update functionality."""
        tracker = ProgressTracker(10, "Test")
        tracker.update(5)
        assert tracker.completed == 5
        
        tracker.update()  # Default increment of 1
        assert tracker.completed == 6
    
    @patch('builtins.print')
    def test_progress_display(self, mock_print):
        """Test progress display output."""
        tracker = ProgressTracker(10, "Test")
        tracker.update(5)
        # Force display by setting last_update to 0
        tracker.last_update = 0
        tracker.update(1)
        
        # Should have called print for progress display
        mock_print.assert_called()


class TestXMLProcessor:
    """Test XML processing functionality."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        config = CLIConfig()
        processor = XMLProcessor(config)
        assert processor.config == config
        assert processor.parser is not None
    
    def test_process_single_file_success(self):
        """Test successful single file processing."""
        config = CLIConfig()
        processor = XMLProcessor(config)
        
        # Create a temporary XML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<?xml version="1.0"?><root><item>test</item></root>')
            xml_path = Path(f.name)
        
        try:
            result = processor.process_single_file(xml_path)
            assert result["success"] is True
            assert result["file"] == str(xml_path)
            assert "confidence" in result
            assert "processing_time_ms" in result
        finally:
            xml_path.unlink()
    
    def test_process_nonexistent_file(self):
        """Test processing non-existent file."""
        config = CLIConfig()
        processor = XMLProcessor(config)
        
        result = processor.process_single_file(Path("nonexistent.xml"))
        assert result["success"] is False
        # For non-existent files, parse_file returns success=False but doesn't raise exception
        # So we should check diagnostics instead of error field
        assert "diagnostics" in result
        assert len(result["diagnostics"]) > 0
    
    def test_find_xml_files_single_file(self):
        """Test finding XML files - single file."""
        config = CLIConfig()
        processor = XMLProcessor(config)
        
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
            xml_path = Path(f.name)
        
        try:
            files = list(processor.find_xml_files(xml_path))
            assert len(files) == 1
            assert files[0] == xml_path
        finally:
            xml_path.unlink()
    
    def test_find_xml_files_directory(self):
        """Test finding XML files in directory."""
        config = CLIConfig()
        processor = XMLProcessor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create XML files
            xml1 = temp_path / "test1.xml"
            xml2 = temp_path / "test2.xml"
            non_xml = temp_path / "test.txt"
            
            xml1.write_text('<root/>')
            xml2.write_text('<root/>')
            non_xml.write_text('not xml')
            
            files = list(processor.find_xml_files(temp_path, recursive=False))
            assert len(files) == 2
            assert xml1 in files
            assert xml2 in files
            assert non_xml not in files
    
    def test_batch_process_single_file(self):
        """Test batch processing with single file."""
        config = CLIConfig()
        processor = XMLProcessor(config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<root><item>test</item></root>')
            xml_path = Path(f.name)
        
        try:
            results = processor.batch_process([xml_path])
            assert len(results) == 1
            assert results[0]["success"] is True
        finally:
            xml_path.unlink()


class TestArgumentParser:
    """Test command-line argument parsing."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_argument_parser()
        assert parser.prog == "ultra-robust-xml"
    
    def test_parse_command_basic(self):
        """Test basic parse command parsing."""
        parser = create_argument_parser()
        args = parser.parse_args(["parse", "test.xml"])
        assert args.command == "parse"
        assert args.paths == [Path("test.xml")]
    
    def test_parse_command_with_options(self):
        """Test parse command with options."""
        parser = create_argument_parser()
        args = parser.parse_args([
            "parse", "test.xml",
            "--format", "csv",
            "--recursive",
            "--preset", "maximum_robustness",
            "--workers", "4"
        ])
        assert args.command == "parse"
        assert args.format == "csv"
        assert args.recursive is True
        assert args.preset == "maximum_robustness"
        assert args.workers == 4
    
    def test_repair_command(self):
        """Test repair command parsing."""
        parser = create_argument_parser()
        args = parser.parse_args(["repair", "test.xml", "--suffix", "_fixed"])
        assert args.command == "repair"
        assert args.suffix == "_fixed"
    
    def test_validate_command(self):
        """Test validate command parsing."""
        parser = create_argument_parser()
        args = parser.parse_args(["validate", "test.xml", "--strict"])
        assert args.command == "validate"
        assert args.strict is True


class TestFormatResults:
    """Test result formatting functions."""
    
    def test_format_json(self):
        """Test JSON formatting."""
        results = [
            {"file": "test.xml", "success": True, "confidence": 0.95},
            {"file": "test2.xml", "success": False, "error": "Parse error"}
        ]
        
        formatted = format_results(results, "json")
        assert isinstance(formatted, str)
        # Should be valid JSON
        parsed = json.loads(formatted)
        assert len(parsed) == 2
    
    def test_format_csv(self):
        """Test CSV formatting."""
        results = [
            {
                "file": "test.xml",
                "success": True,
                "confidence": 0.95,
                "element_count": 5,
                "repair_count": 1,
                "processing_time_ms": 10.5,
                "diagnostics": []
            }
        ]
        
        formatted = format_results(results, "csv")
        lines = formatted.split('\n')
        assert len(lines) == 2  # Header + 1 data line
        assert lines[0].startswith("file,success,confidence")
        assert "test.xml,True,0.950" in lines[1]
    
    def test_format_text(self):
        """Test text formatting."""
        results = [
            {
                "file": "test.xml",
                "success": True,
                "confidence": 0.95,
                "repair_count": 1,
                "processing_time_ms": 10.5,
                "diagnostics": []
            }
        ]
        
        formatted = format_results(results, "text")
        assert "Processed 1 files, 1 successful" in formatted
        assert "test.xml" in formatted
        assert "✓" in formatted
    
    def test_format_empty_results(self):
        """Test formatting empty results."""
        results = []
        
        json_formatted = format_results(results, "json")
        assert json_formatted == "[]"
        
        csv_formatted = format_results(results, "csv")
        assert csv_formatted == ""
        
        text_formatted = format_results(results, "text")
        assert "No results to display" in text_formatted


class TestMainFunction:
    """Test main CLI function."""
    
    def test_main_no_args(self):
        """Test main function with no arguments."""
        exit_code = main([])
        assert exit_code == 1  # Should show help and exit with error
    
    def test_main_unknown_command(self):
        """Test main function with unknown command."""
        # argparse will raise SystemExit for unknown subcommands
        with pytest.raises(SystemExit):
            main(["unknown"])
    
    @patch('ultra_robust_xml_parser.cli.main.cmd_parse')
    def test_main_parse_command(self, mock_cmd_parse):
        """Test main function routing to parse command."""
        mock_cmd_parse.return_value = 0
        
        exit_code = main(["parse", "test.xml"])
        assert exit_code == 0
        mock_cmd_parse.assert_called_once()
    
    @patch('ultra_robust_xml_parser.cli.main.cmd_repair')
    def test_main_repair_command(self, mock_cmd_repair):
        """Test main function routing to repair command."""
        mock_cmd_repair.return_value = 0
        
        exit_code = main(["repair", "test.xml"])
        assert exit_code == 0
        mock_cmd_repair.assert_called_once()
    
    @patch('ultra_robust_xml_parser.cli.main.cmd_validate')
    def test_main_validate_command(self, mock_cmd_validate):
        """Test main function routing to validate command."""
        mock_cmd_validate.return_value = 0
        
        exit_code = main(["validate", "test.xml"])
        assert exit_code == 0
        mock_cmd_validate.assert_called_once()
    
    def test_main_keyboard_interrupt(self):
        """Test main function handling keyboard interrupt."""
        with patch('ultra_robust_xml_parser.cli.main.cmd_parse', side_effect=KeyboardInterrupt):
            exit_code = main(["parse", "test.xml"])
            assert exit_code == 130  # Standard exit code for SIGINT


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_cli_parse_integration(self):
        """Test CLI parse command integration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<?xml version="1.0"?><root><item>test</item></root>')
            xml_path = Path(f.name)
        
        try:
            # Test parsing the file
            exit_code = main(["parse", str(xml_path), "--format", "json"])
            assert exit_code == 0
        finally:
            xml_path.unlink()
    
    def test_cli_validate_integration(self):
        """Test CLI validate command integration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<root><item>valid xml</item></root>')
            xml_path = Path(f.name)
        
        try:
            exit_code = main(["validate", str(xml_path)])
            assert exit_code == 0  # Valid XML should return 0
        finally:
            xml_path.unlink()
    
    def test_cli_repair_integration(self):
        """Test CLI repair command integration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            # Malformed XML that needs repair
            f.write('<root><unclosed>content</root>')
            xml_path = Path(f.name)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                exit_code = main([
                    "repair", str(xml_path),
                    "--output-dir", str(output_dir)
                ])
                assert exit_code == 0
        finally:
            xml_path.unlink()