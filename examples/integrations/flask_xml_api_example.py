"""Example Flask API integration with ultra_robust_xml_parser.

This example demonstrates how to create a REST API that processes XML data
using the ultra_robust_xml_parser with Flask adapter integration.
"""

from flask import Flask, request, Response
from ultra_robust_xml_parser.api import parse, get_adapter

app = Flask(__name__)


@app.route('/xml/parse', methods=['POST'])
def parse_xml():
    """Parse XML data and return various format conversions."""
    try:
        # Get the Flask adapter
        flask_adapter = get_adapter('flask')
        if not flask_adapter:
            return Response(
                '<error>Flask adapter not available</error>', 
                mimetype='application/xml',
                status=500
            )
        
        # Convert Flask request to ParseResult
        conversion_result = flask_adapter.from_target(request)
        
        if not conversion_result.success:
            return Response(
                f'<error>{conversion_result.errors[0] if conversion_result.errors else "Conversion failed"}</error>',
                mimetype='application/xml',
                status=400
            )
        
        parse_result = conversion_result.converted_data
        
        # Prepare response data
        response_data = {
            'success': parse_result.success,
            'confidence': parse_result.confidence,
            'processing_time_ms': conversion_result.conversion_time_ms,
        }
        
        if parse_result.tree and parse_result.tree.root:
            response_data.update({
                'root_tag': parse_result.tree.root.tag,
                'root_text': parse_result.tree.root.text or '',
                'attribute_count': len(parse_result.tree.root.attributes),
                'child_count': len(parse_result.tree.root.children)
            })
        
        # Convert response back to Flask Response
        response_xml = f"""
        <parse_result>
            <success>{response_data['success']}</success>
            <confidence>{response_data['confidence']}</confidence>
            <processing_time_ms>{response_data['processing_time_ms']}</processing_time_ms>
            <root_tag>{response_data.get('root_tag', '')}</root_tag>
            <root_text>{response_data.get('root_text', '')}</root_text>
            <attribute_count>{response_data.get('attribute_count', 0)}</attribute_count>
            <child_count>{response_data.get('child_count', 0)}</child_count>
        </parse_result>
        """.strip()
        
        return Response(response_xml, mimetype='application/xml')
        
    except Exception as e:
        return Response(
            f'<error>Internal server error: {e}</error>',
            mimetype='application/xml',
            status=500
        )


@app.route('/xml/convert/<format>', methods=['POST'])
def convert_xml(format):
    """Convert XML to different formats (pandas, elementtree, etc.)."""
    try:
        # Parse the XML first
        xml_data = request.get_data(as_text=True)
        parse_result = parse(xml_data)
        
        if not parse_result.success:
            return Response(
                '<error>Failed to parse XML</error>',
                mimetype='application/xml',
                status=400
            )
        
        # Get the target adapter
        adapter = get_adapter(format.lower())
        if not adapter:
            available_adapters = ", ".join([
                adapter_meta.name for adapter_meta in 
                get_adapter_registry().list_available_adapters()
            ])
            return Response(
                f'<error>Adapter "{format}" not available. Available: {available_adapters}</error>',
                mimetype='application/xml',
                status=400
            )
        
        # Convert to target format
        conversion_result = adapter.to_target(parse_result)
        
        if not conversion_result.success:
            return Response(
                f'<error>Conversion to {format} failed: {conversion_result.errors[0] if conversion_result.errors else "Unknown error"}</error>',
                mimetype='application/xml',
                status=500
            )
        
        # Return success response
        response_xml = f"""
        <conversion_result>
            <target_format>{format}</target_format>
            <success>true</success>
            <conversion_time_ms>{conversion_result.conversion_time_ms}</conversion_time_ms>
            <metadata>{conversion_result.metadata}</metadata>
        </conversion_result>
        """.strip()
        
        return Response(response_xml, mimetype='application/xml')
        
    except Exception as e:
        return Response(
            f'<error>Internal server error: {e}</error>',
            mimetype='application/xml',
            status=500
        )


@app.route('/xml/validate', methods=['POST'])
def validate_xml():
    """Validate XML structure and provide detailed diagnostics."""
    try:
        xml_data = request.get_data(as_text=True)
        parse_result = parse(xml_data)
        
        # Build validation response
        response_xml = f"""
        <validation_result>
            <valid>{str(parse_result.success).lower()}</valid>
            <confidence>{parse_result.confidence}</confidence>
            <diagnostics_count>{len(parse_result.diagnostics)}</diagnostics_count>
            <warnings_count>{len([d for d in parse_result.diagnostics if d.severity.name == 'WARNING'])}</warnings_count>
            <errors_count>{len([d for d in parse_result.diagnostics if d.severity.name == 'ERROR'])}</errors_count>
        """.strip()
        
        if parse_result.diagnostics:
            response_xml += "\n    <diagnostics>"
            for diagnostic in parse_result.diagnostics[:10]:  # Limit to first 10
                response_xml += f"""
                <diagnostic>
                    <severity>{diagnostic.severity.name}</severity>
                    <message>{diagnostic.message}</message>
                    <component>{diagnostic.component}</component>
                </diagnostic>"""
            response_xml += "\n    </diagnostics>"
        
        response_xml += "\n</validation_result>"
        
        return Response(response_xml, mimetype='application/xml')
        
    except Exception as e:
        return Response(
            f'<error>Internal server error: {e}</error>',
            mimetype='application/xml',
            status=500
        )


@app.route('/xml/benchmark/<adapter_name>', methods=['POST'])
def benchmark_adapter(adapter_name):
    """Benchmark a specific adapter with provided XML data."""
    try:
        from ultra_robust_xml_parser.api import run_performance_benchmark
        
        xml_data = request.get_data(as_text=True)
        
        # Add test data and run benchmark
        benchmark_results = run_performance_benchmark(adapter_name, iterations=10)
        
        if adapter_name not in benchmark_results:
            return Response(
                f'<error>Adapter "{adapter_name}" not available or benchmark failed</error>',
                mimetype='application/xml',
                status=400
            )
        
        results = benchmark_results[adapter_name]
        
        response_xml = f"""
        <benchmark_result>
            <adapter>{adapter_name}</adapter>
            <operations>
        """.strip()
        
        for result in results:
            response_xml += f"""
                <operation type="{result.operation}">
                    <total_operations>{result.total_operations}</total_operations>
                    <average_time_ms>{result.average_time_ms:.2f}</average_time_ms>
                    <operations_per_second>{result.operations_per_second:.2f}</operations_per_second>
                </operation>"""
        
        response_xml += """
            </operations>
        </benchmark_result>"""
        
        return Response(response_xml, mimetype='application/xml')
        
    except Exception as e:
        return Response(
            f'<error>Benchmark error: {e}</error>',
            mimetype='application/xml',
            status=500
        )


# Helper function for adapter registry (simplified)
def get_adapter_registry():
    """Get a simplified adapter registry for listing."""
    from ultra_robust_xml_parser.api import list_available_adapters
    
    class SimpleRegistry:
        def list_available_adapters(self):
            return list_available_adapters()
    
    return SimpleRegistry()


if __name__ == '__main__':
    # Example test data
    test_xml = """
    <books>
        <book id="1">
            <title>The XML Guide</title>
            <author>Jane Doe</author>
            <year>2023</year>
        </book>
        <book id="2">
            <title>Python Programming</title>
            <author>John Smith</author>
            <year>2022</year>
        </book>
    </books>
    """
    
    print("Flask XML API Example")
    print("====================")
    print("Available endpoints:")
    print("  POST /xml/parse - Parse XML and return analysis")
    print("  POST /xml/convert/<format> - Convert XML to specified format")
    print("  POST /xml/validate - Validate XML structure")
    print("  POST /xml/benchmark/<adapter> - Benchmark adapter performance")
    print()
    print("Example test XML:")
    print(test_xml)
    print()
    print("Starting Flask server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)