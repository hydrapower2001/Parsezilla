"""Test module for ultra_robust_xml_parser package initialization."""


def test_package_import() -> None:
    """Test that the package can be imported successfully."""
    # Arrange & Act
    import ultra_robust_xml_parser

    # Assert
    assert ultra_robust_xml_parser is not None


def test_package_has_version() -> None:
    """Test that the package has a version attribute."""
    # Arrange & Act
    import ultra_robust_xml_parser

    # Assert
    assert hasattr(ultra_robust_xml_parser, "__version__")
    assert isinstance(ultra_robust_xml_parser.__version__, str)
    assert ultra_robust_xml_parser.__version__ == "0.1.0"


def test_package_has_author() -> None:
    """Test that the package has an author attribute."""
    # Arrange & Act
    import ultra_robust_xml_parser

    # Assert
    assert hasattr(ultra_robust_xml_parser, "__author__")
    assert isinstance(ultra_robust_xml_parser.__author__, str)
    assert ultra_robust_xml_parser.__author__ == "Ultra Robust XML Parser Team"


def test_package_all_exports() -> None:
    """Test that __all__ contains expected exports."""
    # Arrange & Act
    import ultra_robust_xml_parser

    # Assert
    assert hasattr(ultra_robust_xml_parser, "__all__")
    expected_exports = ["__author__", "__version__"]
    assert ultra_robust_xml_parser.__all__ == expected_exports
