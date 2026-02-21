"""
Doc2Data API Client

Usage:
    python api_client.py <file_path> [--url API_URL] [--format FORMAT]
    python api_client.py --health [--url API_URL]
    
Examples:
    # Health check
    python api_client.py --health --url http://dgx-ip:8000
    
    # Extract any document (auto-detect form type)
    python api_client.py document.pdf --url http://dgx-ip:8000
    
    # Extract UB-04 specifically
    python api_client.py ub04_form.pdf --format ub04
    
    # Extract CMS-1500 specifically
    python api_client.py cms1500_form.pdf --format cms1500
"""
import requests
import json
import sys
import argparse
from pathlib import Path

DEFAULT_API_URL = "http://100.126.216.92:8000"


def health_check(api_url: str):
    """Check API health status."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Status: {result.get('status', 'unknown')}")
            print(f"   Version: {result.get('version', 'unknown')}")
            print(f"   Supported Forms: {', '.join(result.get('supported_forms', []))}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {api_url}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def extract_document(file_path, api_url=DEFAULT_API_URL, format="reducto"):
    """
    Client to call the Doc2Data FastAPI from outside.
    
    Args:
        file_path: Path to PDF or image file
        api_url: API base URL
        format: Output format - reducto, full, ub04, cms1500
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ Error: File {file_path} not found")
        return None

    # Select endpoint based on format
    endpoint_map = {
        "reducto": "/extract/reducto",
        "full": "/extract/v2",
        "ub04": "/extract/ub04",
        "cms1500": "/extract/cms1500",
    }
    endpoint = endpoint_map.get(format, "/extract/v2")
    url = f"{api_url.rstrip('/')}{endpoint}"

    print(f"🚀 Sending {file_path.name} to {url}...")
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            response = requests.post(url, files=files, timeout=120)

        if response.status_code == 200:
            result = response.json()
            output_file = file_path.with_suffix(".json")
            with open(output_file, "w") as out:
                json.dump(result, out, indent=2)
            print(f"✅ Success! Result saved to {output_file}")
            
            # Print summary based on format
            if format == "reducto":
                chunks = result.get("result", {}).get("chunks", [])
                print(f"📊 Extracted {len(chunks)} chunks from {file_path.name}")
            else:
                form_type = result.get("form_type", "unknown")
                fields = result.get("extracted_fields", {})
                business = result.get("business_fields", {})
                filled_business = len([v for v in business.values() if v]) if business else 0
                
                print(f"📋 Form Type: {form_type}")
                print(f"📊 Extracted {len(fields)} raw fields")
                if business:
                    print(f"📊 Mapped to {filled_business} business fields")
                    
                # Print key business fields
                if business:
                    key_fields = ["patient_name", "provider_name", "payer_name", 
                                  "total_charges", "principal_diagnosis"]
                    print("\n🔑 Key Fields:")
                    for k in key_fields:
                        if k in business and business[k]:
                            print(f"   {k}: {business[k]}")
            
            return result
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out. The document may be too large or complex.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Doc2Data API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_client.py --health --url http://dgx-ip:8000
  python api_client.py document.pdf --url http://dgx-ip:8000
  python api_client.py ub04_form.pdf --format ub04
  python api_client.py cms1500_form.pdf --format cms1500 --url http://192.168.1.100:8000
        """
    )
    parser.add_argument("file", nargs="?", help="Path to PDF/Image file")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API Base URL")
    parser.add_argument("--format", choices=["reducto", "full", "ub04", "cms1500"], 
                        default="full", help="Output format/endpoint")
    parser.add_argument("--health", action="store_true", help="Check API health")
    
    args = parser.parse_args()
    
    if args.health:
        success = health_check(args.url)
        sys.exit(0 if success else 1)
    elif args.file:
        result = extract_document(args.file, args.url, args.format)
        sys.exit(0 if result else 1)
    else:
        parser.print_help()
        sys.exit(1)

