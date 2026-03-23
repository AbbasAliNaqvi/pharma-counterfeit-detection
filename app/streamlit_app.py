import sys
import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import build_model

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH   = Path(cfg["paths"]["models"]) / "best_model.pth"
CLASS_NAMES  = cfg["classes"]
IMG_SIZE     = cfg["dataset"]["image_size"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

st.set_page_config(
    page_title="Counterfeit Medicine Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@700;800&display=swap');

  html, body, [class*="css"] {
    background-color: #000000 !important;
    color: #e5e5e5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }

  .main { background-color: #000000 !important; }
  .block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

  h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
  }

  .stButton > button {
    background-color: #10b981 !important;
    color: #000000 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 0px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    width: 100%;
  }
  .stButton > button:hover {
    background-color: #059669 !important;
  }

  .stFileUploader {
    background-color: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
  }

  [data-testid="stFileUploadDropzone"] {
    background-color: #0a0a0a !important;
    border: 1px dashed #333333 !important;
    border-radius: 0 !important;
  }

  .metric-card {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-left: 3px solid #10b981;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
  }

  .metric-label {
    font-size: 0.7rem;
    color: #555555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
  }

  .metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #10b981;
  }

  .result-authentic {
    background: #0a0a0a;
    border: 1px solid #10b981;
    border-left: 4px solid #10b981;
    padding: 1.5rem 2rem;
    text-align: center;
  }

  .result-counterfeit {
    background: #0a0a0a;
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    padding: 1.5rem 2rem;
    text-align: center;
  }

  .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
  }

  .result-sub {
    font-size: 0.75rem;
    color: #666666;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.3rem;
  }

  .section-header {
    font-size: 0.65rem;
    color: #444444;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  .info-block {
    background: #080808;
    border: 1px solid #1a1a1a;
    padding: 1rem 1.2rem;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #888888;
  }

  .tag {
    display: inline-block;
    background: #111111;
    border: 1px solid #222222;
    color: #10b981;
    font-size: 0.65rem;
    padding: 0.2rem 0.5rem;
    letter-spacing: 0.08em;
    margin-right: 0.3rem;
    text-transform: uppercase;
  }

  div[data-testid="stImage"] img {
    border: 1px solid #1a1a1a;
  }

  .stProgress > div > div {
    background-color: #10b981 !important;
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model, device = build_model(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(pil_img).unsqueeze(0)


class GradCAM:
    def __init__(self, model, target_layer):
        self.activations = None
        self.gradients   = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )
        self.model = model

    def generate(self, inp: torch.Tensor, class_idx: int = None):
        self.model.eval()
        out = self.model(inp)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def make_heatmap_overlay(tensor: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img  = tensor.squeeze(0).numpy() * std + mean
    img  = np.clip(img, 0, 1)
    rgb  = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

    hm_u8    = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    hm_rgb   = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    overlay = (0.45 * hm_rgb + 0.55 * rgb).astype(np.uint8)
    return rgb, overlay


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#000000")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def confidence_gauge(confidence: float, label: str) -> str:
    pct     = confidence * 100
    color   = "#10b981" if label == "Authentic" else "#ef4444"
    radius  = 54
    circ    = 2 * 3.14159 * radius
    dash    = circ * confidence
    gap     = circ - dash
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:1.2rem 0;">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r="{radius}" fill="none"
                stroke="#111111" stroke-width="10"/>
        <circle cx="70" cy="70" r="{radius}" fill="none"
                stroke="{color}" stroke-width="10"
                stroke-dasharray="{dash:.1f} {gap:.1f}"
                stroke-dashoffset="{circ/4:.1f}"
                stroke-linecap="butt"
                transform="rotate(0 70 70)"/>
        <text x="70" y="65" text-anchor="middle"
              font-family="IBM Plex Mono,monospace"
              font-size="18" font-weight="600" fill="{color}">{pct:.1f}%</text>
        <text x="70" y="83" text-anchor="middle"
              font-family="IBM Plex Mono,monospace"
              font-size="9" fill="#555555" letter-spacing="2">CONFIDENCE</text>
      </svg>
    </div>
    """


st.markdown("""
<div style="border-bottom:1px solid #111111;padding-bottom:1.5rem;margin-bottom:2rem;">
  <div style="font-size:0.65rem;color:#444;letter-spacing:0.25em;text-transform:uppercase;
              margin-bottom:0.4rem;">AI-Based Detection System</div>
  <h1 style="margin:0;font-size:2.4rem;color:#ffffff;letter-spacing:-0.03em;
             font-family:Syne,sans-serif;">
    Counterfeit Medicine Detector
  </h1>
  <div style="margin-top:0.6rem;">
    <span class="tag">MobileNetV3-Small</span>
    <span class="tag">Grad-CAM</span>
    <span class="tag">96% Accuracy</span>
    <span class="tag">AUC 0.9947</span>
  </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a medicine image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, use_column_width=True)
        st.markdown(f"""
        <div style="font-size:0.7rem;color:#444;margin-top:0.5rem;">
          {uploaded.name} &nbsp;|&nbsp; {pil_img.size[0]}x{pil_img.size[1]}px
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("RUN ANALYSIS", disabled=uploaded is None)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
      <b style="color:#e5e5e5;">Backbone</b><br>MobileNetV3-Small<br>
      pretrained on ImageNet, fine-tuned on synthetic<br>
      pharmaceutical dataset.<br><br>
      <b style="color:#e5e5e5;">Dataset</b><br>8,000 images — 4,000 authentic<br>
      + 4,000 synthetically degraded counterfeits.<br><br>
      <b style="color:#e5e5e5;">Explainability</b><br>Grad-CAM highlights which<br>
      image regions drove the classification.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">Analysis Results</div>',
                unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div style="border:1px dashed #1a1a1a;padding:4rem 2rem;
                    text-align:center;color:#333333;">
          <div style="font-size:0.75rem;letter-spacing:0.15em;text-transform:uppercase;">
            Upload an image to begin analysis
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif run_btn or ("last_result" in st.session_state and
                     st.session_state.get("last_file") == uploaded.name):

        with st.spinner("Running inference..."):
            model, device = load_model()
            inp = preprocess(pil_img).to(device)

            gradcam     = GradCAM(model, model.features[12])
            with torch.enable_grad():
                heatmap, pred_idx = gradcam.generate(inp)
            gradcam.remove()

            with torch.no_grad():
                logits = model(inp)
                probs  = torch.softmax(logits, dim=1)[0]

            pred_label  = CLASS_NAMES[pred_idx]
            confidence  = probs[pred_idx].item()
            auth_prob   = probs[0].item()
            fake_prob   = probs[1].item()

            rgb_img, overlay = make_heatmap_overlay(inp.cpu(), heatmap)

            st.session_state["last_result"] = {
                "pred_label": pred_label,
                "confidence": confidence,
                "auth_prob":  auth_prob,
                "fake_prob":  fake_prob,
                "rgb_img":    rgb_img,
                "overlay":    overlay,
            }
            st.session_state["last_file"] = uploaded.name

        result = st.session_state["last_result"]
        pred_label = result["pred_label"]
        confidence = result["confidence"]

        result_class = "result-authentic" if pred_label == "Authentic" else "result-counterfeit"
        label_color  = "#10b981" if pred_label == "Authentic" else "#ef4444"
        verdict      = "AUTHENTIC" if pred_label == "Authentic" else "COUNTERFEIT"

        st.markdown(f"""
        <div class="{result_class}">
          <div class="result-label" style="color:{label_color};">{verdict}</div>
          <div class="result-sub">Classification Result</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Confidence</div>
              <div class="metric-value">{confidence*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#10b981;">
              <div class="metric-label">Authentic prob</div>
              <div class="metric-value" style="color:#10b981;">{result['auth_prob']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#ef4444;">
              <div class="metric-label">Counterfeit prob</div>
              <div class="metric-value" style="color:#ef4444;">{result['fake_prob']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Grad-CAM Explainability</div>',
                    unsafe_allow_html=True)

        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(result["rgb_img"], caption="Original (224x224)", use_column_width=True)
        with gc2:
            st.image(result["overlay"], caption="Grad-CAM Overlay", use_column_width=True)

        st.markdown(f"""
        <div class="info-block" style="margin-top:1rem;">
          The heatmap highlights regions the model used to classify this image as
          <b style="color:{label_color};">{verdict}</b>.
          Red/yellow regions indicate high activation — areas with the strongest
          influence on the decision. Blue regions had low influence.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Confidence Gauge</div>',
                    unsafe_allow_html=True)
        st.markdown(
            confidence_gauge(confidence, pred_label),
            unsafe_allow_html=True
        )

st.markdown("""
<div style="border-top:1px solid #111111;margin-top:3rem;padding-top:1rem;
            font-size:0.65rem;color:#333333;text-align:center;letter-spacing:0.1em;">
  COUNTERFEIT MEDICINE DETECTOR &nbsp;|&nbsp; MobileNetV3-Small &nbsp;|&nbsp;
  TEST ACC 96.00% &nbsp;|&nbsp; AUC-ROC 0.9947
</div>
""", unsafe_allow_html=True)