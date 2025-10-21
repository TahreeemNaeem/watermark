from typing import Any, Dict, List, Tuple
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from transformers import OwlViTVisionModel
from torch import nn
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# OWLv2 classification head
class DetectorModelOwl(nn.Module):
    owl: OwlViTVisionModel

    def __init__(self, model_path: str, dropout: float, n_hidden: int = 768):
        super().__init__()
        owl = OwlViTVisionModel.from_pretrained(model_path)
        assert isinstance(owl, OwlViTVisionModel)
        self.owl = owl
        self.owl.requires_grad_(False)
        self.transforms = None
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_hidden, eps=1e-5)
        self.linear1 = nn.Linear(n_hidden, n_hidden * 2)
        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(n_hidden * 2, eps=1e-5)
        self.linear2 = nn.Linear(n_hidden * 2, 2)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)
            x = outputs.last_hidden_state
            x = self.dropout1(x)
            x = self.ln1(x)
            x = self.linear1(x)
            x = self.act1(x)
            x = self.dropout2(x)
            x, _ = x.max(dim=1)
            x = self.ln2(x)
            x = self.linear2(x)
            if labels is not None:
                loss = F.cross_entropy(x, labels)
                return (x, loss)
            return (x,)


def owl_predict(image: Image.Image, owl_model) -> bool:
    """Classify if image contains watermark using OWLv2"""
    big_side = max(image.size)
    new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
    new_image.paste(image, (0, 0))
    preped = new_image.resize((960, 960), Image.BICUBIC)
    preped = TVF.pil_to_tensor(preped)
    preped = preped / 255.0
    input_image = TVF.normalize(
        preped,
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    )
    logits, = owl_model(input_image.to('cpu').unsqueeze(0), None)
    probs = F.softmax(logits, dim=1)
    prediction = torch.argmax(probs.cpu(), dim=1)
    return prediction.item() == 1


def yolo_detect_boxes(image: Image.Image, yolo_model, conf_threshold: float) -> List[List[int]]:
    """Detect watermark bounding boxes using YOLO"""
    results = yolo_model(image, imgsz=1024, augment=True, iou=0.5, conf=conf_threshold)
    assert len(results) == 1
    result = results[0]
    
    boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # Get coordinates in xyxy format
            xyxy = box.xyxy[0].cpu().numpy()
            boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
    
    return boxes


def sam2_create_masks(image: np.ndarray, boxes: List[List[int]], sam_model, model_choice: str) -> np.ndarray:
    """Create precise masks for detected boxes using SAM2"""
    if not boxes:
        return image
    
    predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(image)
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(boxes),
        multimask_output=False,
    )
    
    multi_box = len(scores) > 1
    return show_masks(
        image=image,
        masks=masks,
        scores=scores if len(scores) == 1 else None,
        only_best=not multi_box,
    )


def create_visualization(image: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
    """Create visualization with bounding boxes"""
    vis_image = image.copy()
    for box in boxes:
        cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return vis_image


def predict(image: Image.Image, sam_model_choice: str, conf_threshold: float, 
            owl_model, yolo_model):
    """Main prediction function combining all models"""
    if image is None:
        return None, None, "Please upload an image"
    
    # Convert PIL to numpy
    image_np = np.array(image)
    if image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Step 1: OWLv2 Classification
    has_watermark = owl_predict(image, owl_model)
    owl_result = "Watermarked ✓" if has_watermark else "Not Watermarked ✗"
    
    # Step 2: YOLO Detection
    boxes = yolo_detect_boxes(image, yolo_model, conf_threshold)
    
    if not boxes:
        return image_np, None, f"OWLv2: {owl_result}\nYOLO: No watermarks detected"
    
    # Step 3: Create visualization with boxes
    box_image = create_visualization(image_np, boxes)
    
    # Step 4: SAM2 Masking
    config_mapping = {
        "tiny": "sam2_hiera_t.yaml",
        "small": "sam2_hiera_s.yaml", 
        "base_plus": "sam2_hiera_b+.yaml",
        "large": "sam2_hiera_l.yaml"
    }
    
    config_name = config_mapping.get(sam_model_choice, "sam2_hiera_t.yaml")
    sam2_model = build_sam2(
        config_file=f"assets/configs/{config_name}",
        ckpt_path=f"assets/checkpoints/sam2_hiera_{sam_model_choice}.pt",
        device="cpu",
    )
    
    mask_image = sam2_create_masks(image_np, boxes, sam2_model, sam_model_choice)
    
    result_text = f"OWLv2: {owl_result}\nYOLO: Detected {len(boxes)} watermark(s)"
    
    return box_image, mask_image, result_text


# Initialize models
print("Loading OWLv2 model...")
owl_model = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0)
# Uncomment when you have the checkpoint:
# owl_model.load_state_dict(torch.load("far5y1y5-8000.pt", map_location="cpu"))
owl_model.eval()

print("Loading YOLO model...")
# Uncomment when you have the checkpoint:
# yolo_model = YOLO("yolo11x-train28-best.pt")
# For demo purposes, using a pretrained YOLO model:
yolo_model = YOLO("yolov8n.pt")

# Create Gradio Interface
with gr.Blocks(title="Watermark Detection & Masking") as demo:
    gr.HTML(
        """
        <h1 style="text-align: center;">🎯 Watermark Detection & Masking Pipeline</h1>
        <p style="text-align: center;">
            Combines OWLv2 classification, YOLO detection, and SAM2 segmentation
        </p>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            input_image = gr.Image(type="pil", label="Upload Image")
            
            gr.Markdown("### ⚙️ Settings")
            sam_model = gr.Dropdown(
                choices=["tiny", "small", "base_plus", "large"],
                value="tiny",
                label="SAM2 Model",
                info="Larger models are more accurate but slower"
            )
            conf_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="YOLO Confidence Threshold"
            )
            
            detect_btn = gr.Button("🔍 Detect & Mask Watermarks", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            result_text = gr.Textbox(
                label="Detection Summary",
                lines=3,
                interactive=False
            )
            
            with gr.Row():
                with gr.Column():
                    box_output = gr.Image(label="YOLO Detections (Boxes)")
                with gr.Column():
                    mask_output = gr.Image(label="SAM2 Segmentation (Masks)")
    
    gr.Markdown(
        """
        ### 📝 Pipeline Steps:
        1. **OWLv2 Classification**: Determines if image contains watermarks
        2. **YOLO Detection**: Locates watermark positions with bounding boxes
        3. **SAM2 Segmentation**: Creates precise pixel-level masks for each watermark
        
        ### 💡 Tips:
        - Lower confidence threshold detects more potential watermarks (may include false positives)
        - Higher confidence threshold only shows highly confident detections
        - Larger SAM2 models provide better segmentation quality
        """
    )
    
    detect_btn.click(
        fn=lambda img, sam, conf: predict(img, sam, conf, owl_model, yolo_model),
        inputs=[input_image, sam_model, conf_threshold],
        outputs=[box_output, mask_output, result_text]
    )

if __name__ == "__main__":
    demo.launch()