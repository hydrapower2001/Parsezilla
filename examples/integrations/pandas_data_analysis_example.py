"""Example pandas integration for XML data analysis.

This example demonstrates how to use the pandas adapter to convert XML data
into DataFrames for analysis and visualization.
"""

import pandas as pd
from ultra_robust_xml_parser.api import parse, get_adapter


def analyze_xml_with_pandas():
    """Example analysis of XML data using pandas integration."""
    
    # Sample XML data representing sales data
    sales_xml = """
    <sales_data>
        <sale id="1" date="2023-01-15">
            <product>Laptop</product>
            <category>Electronics</category>
            <price>1200.00</price>
            <quantity>2</quantity>
            <customer>John Doe</customer>
        </sale>
        <sale id="2" date="2023-01-16">
            <product>Mouse</product>
            <category>Electronics</category>
            <price>25.00</price>
            <quantity>5</quantity>
            <customer>Jane Smith</customer>
        </sale>
        <sale id="3" date="2023-01-16">
            <product>Keyboard</product>
            <category>Electronics</category>
            <price>75.00</price>
            <quantity>3</quantity>
            <customer>Bob Johnson</customer>
        </sale>
        <sale id="4" date="2023-01-17">
            <product>Desk</product>
            <category>Furniture</category>
            <price>300.00</price>
            <quantity>1</quantity>
            <customer>Alice Brown</customer>
        </sale>
    </sales_data>
    """
    
    print("XML Data Analysis with Pandas Integration")
    print("=" * 45)
    
    # Parse the XML
    print("1. Parsing XML data...")
    parse_result = parse(sales_xml)
    
    if not parse_result.success:
        print("Failed to parse XML!")
        return
    
    print(f"   ✓ Parsing successful (confidence: {parse_result.confidence:.2f})")
    
    # Get pandas adapter
    print("\n2. Getting pandas adapter...")
    pandas_adapter = get_adapter('pandas')
    
    if not pandas_adapter:
        print("   ✗ Pandas adapter not available!")
        print("   Make sure pandas is installed: pip install pandas")
        return
    
    print("   ✓ Pandas adapter available")
    
    # Convert to DataFrame
    print("\n3. Converting XML to DataFrame...")
    conversion_result = pandas_adapter.to_target(parse_result)
    
    if not conversion_result.success:
        print(f"   ✗ Conversion failed: {conversion_result.errors}")
        return
    
    df = conversion_result.converted_data
    print(f"   ✓ Conversion successful ({conversion_result.conversion_time_ms:.2f}ms)")
    print(f"   DataFrame shape: {df.shape}")
    
    # Display the DataFrame
    print("\n4. DataFrame Overview:")
    print("-" * 40)
    print(df.head())
    
    print(f"\nDataFrame Info:")
    print(f"- Shape: {df.shape}")
    print(f"- Columns: {list(df.columns)}")
    print(f"- Data types: {dict(df.dtypes)}")
    
    # Extract business data for analysis
    print("\n5. Extracting Business Data...")
    
    # Filter for sales data rows (those with product information)
    sales_df = df[df['tag'] == 'sale'].copy()
    
    # Extract attributes into separate columns
    if not sales_df.empty:
        sales_df['sale_id'] = sales_df['attr_id']
        sales_df['sale_date'] = sales_df['attr_date']
        
        print(f"   Found {len(sales_df)} sales records")
        print("\nSales Overview:")
        print(sales_df[['sale_id', 'sale_date', 'path']].head())
    
    # Get product details
    product_df = df[df['tag'] == 'product'].copy()
    category_df = df[df['tag'] == 'category'].copy()
    price_df = df[df['tag'] == 'price'].copy()
    quantity_df = df[df['tag'] == 'quantity'].copy()
    customer_df = df[df['tag'] == 'customer'].copy()
    
    print(f"\n6. Business Analysis:")
    print("-" * 30)
    
    if not product_df.empty:
        print(f"Products sold: {len(product_df)}")
        print("Product list:")
        for product in product_df['text'].values:
            print(f"  - {product}")
    
    if not category_df.empty:
        print(f"\nCategories:")
        category_counts = category_df['text'].value_counts()
        for category, count in category_counts.items():
            print(f"  - {category}: {count} items")
    
    if not price_df.empty:
        prices = pd.to_numeric(price_df['text'], errors='coerce')
        print(f"\nPrice Analysis:")
        print(f"  - Average price: ${prices.mean():.2f}")
        print(f"  - Min price: ${prices.min():.2f}")
        print(f"  - Max price: ${prices.max():.2f}")
    
    if not quantity_df.empty:
        quantities = pd.to_numeric(quantity_df['text'], errors='coerce')
        total_quantity = quantities.sum()
        print(f"\nQuantity Analysis:")
        print(f"  - Total items sold: {total_quantity}")
        print(f"  - Average quantity per sale: {quantities.mean():.1f}")
    
    # Revenue calculation
    if not price_df.empty and not quantity_df.empty:
        prices = pd.to_numeric(price_df['text'], errors='coerce')
        quantities = pd.to_numeric(quantity_df['text'], errors='coerce')
        revenues = prices * quantities
        total_revenue = revenues.sum()
        
        print(f"\nRevenue Analysis:")
        print(f"  - Total revenue: ${total_revenue:.2f}")
        print(f"  - Average revenue per sale: ${revenues.mean():.2f}")
    
    # Demonstrate reverse conversion
    print(f"\n7. Testing Reverse Conversion...")
    
    # Create a simplified DataFrame for reverse conversion
    simple_df = pd.DataFrame({
        'tag': ['summary', 'total_sales', 'total_revenue'],
        'text': [f'{len(sales_df)} sales processed', str(len(product_df)), f'${total_revenue:.2f}' if 'total_revenue' in locals() else '0'],
        'path': ['/', '/summary[0]', '/summary[1]']
    })
    
    reverse_result = pandas_adapter.from_target(simple_df)
    
    if reverse_result.success:
        print(f"   ✓ Reverse conversion successful ({reverse_result.conversion_time_ms:.2f}ms)")
        reverse_parse_result = reverse_result.converted_data
        print(f"   Reverse parse success: {reverse_parse_result.success}")
    else:
        print(f"   ✗ Reverse conversion failed: {reverse_result.errors}")
    
    return df


def create_sample_report(df):
    """Create a sample analysis report from the DataFrame."""
    
    print("\n8. Sample Analysis Report")
    print("=" * 30)
    
    # Group data by element types for better analysis
    element_summary = df.groupby('tag').size().sort_values(ascending=False)
    
    print("XML Structure Analysis:")
    print("-" * 22)
    for element, count in element_summary.items():
        print(f"{element:<15}: {count} occurrences")
    
    # Analyze paths to understand document structure
    print(f"\nDocument Structure Depth:")
    print("-" * 25)
    max_depth = df['path'].str.count('/').max()
    print(f"Maximum nesting depth: {max_depth}")
    
    # Find most common parent elements
    parent_paths = df['path'].str.extract(r'(.+)/[^/]+$')[0].dropna().value_counts().head(5)
    if not parent_paths.empty:
        print(f"\nMost common parent paths:")
        for path, count in parent_paths.items():
            print(f"  {path}: {count} children")


if __name__ == "__main__":
    # Run the analysis
    try:
        df = analyze_xml_with_pandas()
        if df is not None:
            create_sample_report(df)
            
            print(f"\n" + "=" * 50)
            print("Analysis completed successfully!")
            print("This example demonstrates:")
            print("• XML to pandas DataFrame conversion")
            print("• Business data extraction from XML")
            print("• Statistical analysis with pandas")
            print("• Reverse conversion capabilities")
            
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install pandas")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()