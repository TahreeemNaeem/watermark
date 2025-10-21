from PIL import Image
from check import WatermarkRemovalPipeline

# Initialize
pipeline = WatermarkRemovalPipeline(
    sam_model="large",
    inpaint_method="lama"
)

# Process single image
image = Image.open("image.png")
result = pipeline.process(image)

if result["success"]:
    Image.fromarray(result["result"]).save("clean.jpg")
    print(f"Removed {result['watermarks_found']} watermarks")

# # Batch processing
# results = pipeline.batch_process(
#     image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
#     output_dir="cleaned_output"
# )

# Get statistics
stats = pipeline.get_stats()
print(f"Processed: {stats['processed']}, Errors: {stats['errors']}")