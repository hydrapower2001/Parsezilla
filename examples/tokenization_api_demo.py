#!/usr/bin/env python3
"""Demonstration of the ultra-robust XML tokenization API.

This example shows how to use the comprehensive tokenization API with
various configuration options and demonstrates the rich result objects.
"""

from ultra_robust_xml_parser.character import CharacterStreamResult
from ultra_robust_xml_parser.character.encoding import EncodingResult, DetectionMethod
from ultra_robust_xml_parser.shared import TokenizationConfig
from ultra_robust_xml_parser.tokenization.api import (
    EnhancedXMLTokenizer,
    StreamingTokenizer,
    TokenFilter,
)


def create_sample_xml():
    """Create sample XML content for demonstration."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>API Demo Document</title>
        <author>Ultra-Robust XML Parser</author>
    </metadata>
    <content>
        <section id="intro">
            <heading>Introduction</heading>
            <paragraph>This demonstrates the comprehensive tokenization API.</paragraph>
        </section>
        <section id="features">
            <heading>Features</heading>
            <list>
                <item>Never-fail philosophy</item>
                <item>Rich result objects</item>
                <item>Streaming support</item>
                <item>Token filtering</item>
                <item>Performance benchmarking</item>
            </list>
        </section>
    </content>
</document>'''


def create_character_stream(xml_content: str) -> CharacterStreamResult:
    """Create a character stream result for the XML content."""
    encoding_result = EncodingResult(
        encoding="utf-8",
        confidence=1.0,
        method=DetectionMethod.FALLBACK,
        issues=[]
    )
    
    return CharacterStreamResult(
        text=xml_content,
        encoding=encoding_result,
        confidence=1.0,
        processing_time=0.0,
        diagnostics=[]
    )


def demo_basic_tokenization():
    """Demonstrate basic enhanced tokenization."""
    print("🔷 Basic Enhanced Tokenization Demo")
    print("=" * 50)
    
    # Create XML content and character stream
    xml_content = create_sample_xml()
    char_stream = create_character_stream(xml_content)
    
    # Create tokenizer with default configuration
    tokenizer = EnhancedXMLTokenizer()
    
    # Tokenize the content
    result = tokenizer.tokenize(char_stream)
    
    # Display results
    print(f"✅ Success: {result.success}")
    print(f"🔢 Token Count: {result.token_count}")
    print(f"⭐ Confidence: {result.confidence:.3f}")
    print(f"⏱️  Processing Time: {result.performance.processing_time_ms:.2f}ms")
    print(f"📊 Characters/Second: {result.performance.characters_per_second:.0f}")
    
    # Show token type distribution
    print("\n📈 Token Type Distribution:")
    for token_type, count in result.metadata.token_type_distribution.items():
        print(f"   {token_type}: {count}")
    
    # Show summary
    print(f"\n📋 Summary: {result.summary()}")
    print()


def demo_configuration_presets():
    """Demonstrate different configuration presets."""
    print("🔷 Configuration Presets Demo")
    print("=" * 50)
    
    xml_content = create_sample_xml()
    char_stream = create_character_stream(xml_content)
    
    configs = [
        ("Conservative", TokenizationConfig.conservative()),
        ("Balanced", TokenizationConfig.balanced()),
        ("Aggressive", TokenizationConfig.aggressive()),
        ("Performance", TokenizationConfig.performance_optimized()),
    ]
    
    for config_name, config in configs:
        tokenizer = EnhancedXMLTokenizer(config)
        result = tokenizer.tokenize(char_stream)
        
        print(f"{config_name:12} | "
              f"Tokens: {result.token_count:3d} | "
              f"Confidence: {result.confidence:.3f} | "
              f"Time: {result.performance.processing_time_ms:5.2f}ms")
    
    print()


def demo_streaming_tokenization():
    """Demonstrate streaming tokenization."""
    print("🔷 Streaming Tokenization Demo")
    print("=" * 50)
    
    # Create larger XML content
    large_xml_parts = ['<?xml version="1.0"?>', '<large_document>']
    for i in range(20):
        large_xml_parts.append(f'    <item id="{i}">Content for item {i}</item>')
    large_xml_parts.append('</large_document>')
    
    large_xml = '\n'.join(large_xml_parts)
    char_stream = create_character_stream(large_xml)
    
    # Configure streaming
    config = TokenizationConfig.performance_optimized()
    config.streaming.chunk_size = 200
    
    streaming_tokenizer = StreamingTokenizer(config, "streaming-demo")
    
    # Track progress
    progress_updates = []
    def progress_callback(progress: float, token_count: int):
        progress_updates.append((progress, token_count))
        if len(progress_updates) % 5 == 0:  # Show every 5th update
            print(f"   📈 Progress: {progress:.1%}, Tokens: {token_count}")
    
    # Stream tokens
    print("🚀 Starting streaming tokenization...")
    tokens = []
    token_generator = streaming_tokenizer.tokenize_stream(char_stream, progress_callback)
    
    for token in token_generator:
        tokens.append(token)
    
    print(f"✅ Streaming completed!")
    print(f"🔢 Total tokens streamed: {len(tokens)}")
    print(f"📊 Progress updates: {len(progress_updates)}")
    print()


def demo_token_filtering():
    """Demonstrate token filtering capabilities."""
    print("🔷 Token Filtering Demo")  
    print("=" * 50)
    
    xml_content = create_sample_xml()
    char_stream = create_character_stream(xml_content)
    
    # Tokenize without filtering
    tokenizer = EnhancedXMLTokenizer()
    result = tokenizer.tokenize(char_stream)
    
    print(f"📊 Original token count: {result.token_count}")
    
    # Create token filter
    token_filter = TokenFilter()
    
    # Filter by token types
    from ultra_robust_xml_parser.tokenization import TokenType
    text_tokens = token_filter.filter_by_type(
        result.tokens, 
        {TokenType.TEXT, TokenType.TAG_NAME}
    )
    print(f"🔤 Text + Tag Name tokens: {len(text_tokens)}")
    
    # Filter by confidence  
    high_confidence_tokens = token_filter.filter_by_confidence(result.tokens, 0.9)
    print(f"⭐ High confidence tokens (≥0.9): {len(high_confidence_tokens)}")
    
    # Apply multiple filters
    filtered_tokens = token_filter.apply_filters(
        result.tokens,
        type_filter={TokenType.TEXT, TokenType.TAG_NAME, TokenType.ATTR_VALUE},
        confidence_threshold=0.8
    )
    print(f"🎯 Multi-filtered tokens: {len(filtered_tokens)}")
    
    # Show some sample tokens
    print(f"\n📝 Sample filtered tokens:")
    for token in filtered_tokens[:5]:
        print(f"   {token.type.name}: '{token.value[:30]}...' (confidence: {token.confidence:.3f})")
    
    print()


def demo_error_handling():
    """Demonstrate error handling and diagnostics."""
    print("🔷 Error Handling & Diagnostics Demo")
    print("=" * 50)
    
    # Create malformed XML
    malformed_xml = '''<?xml version="1.0"?>
<root>
    <unclosed_tag>
    <element attr=unquoted>Content
    <element attr="unclosed>More content</element>
    <!-- Unclosed comment
    <empty
</root>'''
    
    char_stream = create_character_stream(malformed_xml)
    
    # Use aggressive recovery configuration
    config = TokenizationConfig.aggressive()
    tokenizer = EnhancedXMLTokenizer(config)
    
    result = tokenizer.tokenize(char_stream)
    
    print(f"✅ Success (never-fail): {result.success}")
    print(f"🔢 Token Count: {result.token_count}")
    print(f"⭐ Confidence: {result.confidence:.3f}")
    print(f"❌ Error Rate: {result.error_rate:.3f}")
    print(f"🔧 Repair Rate: {result.repair_rate:.3f}")
    
    # Show diagnostics
    print(f"\n🩺 Diagnostics ({len(result.diagnostics)}):")
    for diag in result.diagnostics[:3]:  # Show first 3
        print(f"   [{diag.severity.name}] {diag.component}: {diag.message}")
    
    # Show error recovery statistics
    print(f"\n🔄 Recovery Statistics:")
    print(f"   Recovered tokens: {result.metadata.recovered_tokens}")
    print(f"   Synthetic tokens: {result.metadata.synthetic_tokens}")
    print(f"   Error tokens: {result.metadata.error_tokens}")
    
    print()


def main():
    """Run all demonstration examples."""
    print("🚀 Ultra-Robust XML Tokenization API Demo")
    print("=" * 60)
    print()
    
    try:
        demo_basic_tokenization()
        demo_configuration_presets()
        demo_streaming_tokenization()
        demo_token_filtering()
        demo_error_handling()
        
        print("🎉 All demonstrations completed successfully!")
        print("   The tokenization API is ready for production use.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()