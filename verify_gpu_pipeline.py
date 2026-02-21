
import sys
import os
import asyncio
import json
from src.pipelines.multi_agent_pipeline import MultiAgentPipeline, PipelineConfig

async def main():
    print("🧪 Verifying GPU Pipeline...")
    
    # Check GPU availability (via OpenCV/Paddle)
    import cv2
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"   OpenCV CUDA Devices: {count}")
    except:
        print("   OpenCV CUDA not built/available")

    # Initialize Pipeline
    config = PipelineConfig(
        enable_form_detection=True,
        enable_alignment=True,
        enable_trocr=True, # Will fallback to CPU if GPU not avail
        enable_slm_labeling=False, # Skip SLM for quick verification
        enable_vlm_figures=False
    )
    pipeline = MultiAgentPipeline(config)
    
    # Test on a sample file
    sample_path = "data/sample_docs/cms1500.pdf"
    if not os.path.exists(sample_path):
        print(f"⚠️ Sample file not found at {sample_path}, using dummy check.")
        return

    print(f"   Processing {sample_path}...")
    try:
        result = await pipeline.process(sample_path)
        
        if result.get("success"):
            print("✅ Pipeline Success!")
            print(f"   Fields Extracted: {len(result.get('extracted_fields', {}))}")
            print(f"   Form Type: {result.get('form_type')}")
            
            # Check Reducto format
            if "reducto_format" in result:
                 print("✅ Reducto format present")
                 chunks = result["reducto_format"]["result"]["chunks"]
                 print(f"   Chunks: {len(chunks)}")
                 if chunks and chunks[0].get("blocks"):
                     print(f"   Blocks in first chunk: {len(chunks[0]['blocks'])}")
            else:
                 print("❌ Reducto format MISSING")
            
            # Print field details sample to check layout detection source
            if result.get("field_details"):
                sample = result["field_details"][0]
                print(f"   Sample Field Source: {sample.get('detected_by', 'unknown')}")
                 
            # Save output
            with open("verify_result.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
        else:
            print("❌ Pipeline failed (success=False)")
            
    except Exception as e:
        print(f"❌ Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

