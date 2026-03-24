import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import build_model

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH    = Path(cfg["paths"]["models"]) / "best_model.pth"
CLASS_NAMES   = cfg["classes"]
IMG_SIZE      = cfg["dataset"]["image_size"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

st.set_page_config(page_title="PharmaGuard", page_icon="", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@800&display=swap');

html,body,[class*="css"]{font-family:'IBM Plex Mono',monospace!important;background:#000!important;color:#ccc!important}
.block-container{max-width:700px!important;padding:1.5rem 1rem!important}
#MainMenu,footer,header,.stDeployButton{display:none!important}

.stTabs [data-baseweb="tab-list"]{gap:0;background:#000;border-bottom:1px solid #111}
.stTabs [data-baseweb="tab"]{background:#000!important;color:#333!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;padding:.6rem 1.2rem;border:none;border-radius:0}
.stTabs [aria-selected="true"]{color:#10b981!important;border-bottom:2px solid #10b981!important}

.stButton>button{background:#10b981!important;color:#000!important;font-family:'IBM Plex Mono',monospace!important;
  font-weight:600!important;font-size:.78rem!important;border:none!important;border-radius:0!important;
  padding:.65rem!important;letter-spacing:.1em;text-transform:uppercase;width:100%!important}
.stButton>button:hover{background:#059669!important}
.stButton>button:disabled{background:#111!important;color:#333!important}

[data-testid="stFileUploadDropzone"]{background:#080808!important;border:1px dashed #1a1a1a!important;border-radius:0!important}
[data-testid="stImage"] img{border:1px solid #111;width:100%}

.verdict{font-family:'Syne',sans-serif!important;font-size:2.2rem;font-weight:800;
  letter-spacing:-.03em;text-align:center;padding:1.2rem;border:1px solid #10b981;margin-bottom:.8rem}
.verdict-r{border-color:#ef4444!important}
.kpi{background:#060606;border:1px solid #0f0f0f;border-top:2px solid #10b981;padding:.9rem 1rem;margin-bottom:.4rem}
.kpi-r{border-top-color:#ef4444!important}
.kpi-l{font-size:.58rem;color:#444;text-transform:uppercase;letter-spacing:.15em;margin-bottom:.2rem}
.kpi-v{font-size:1.3rem;font-weight:600;color:#10b981}
.kpi-vr{font-size:1.3rem;font-weight:600;color:#ef4444}
.hd{font-size:.58rem;color:#333;text-transform:uppercase;letter-spacing:.2em;
  border-bottom:1px solid #0a0a0a;padding-bottom:.3rem;margin:1rem 0 .7rem}
.tip{background:#050505;border:1px solid #0f0f0f;padding:.8rem 1rem;
  font-size:.72rem;line-height:1.8;color:#555;margin-bottom:.8rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cpu")
    model, device = build_model(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def preprocess(pil_img):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(pil_img).unsqueeze(0)


def analyse(pil_img):
    model, device = load_model()
    inp = preprocess(pil_img).to(device)

    gc_obj = type("GC", (), {"act": None, "grad": None})()
    fh = model.features[12].register_forward_hook(
        lambda m, i, o: setattr(gc_obj, "act", o.detach()))
    bh = model.features[12].register_full_backward_hook(
        lambda m, gi, go: setattr(gc_obj, "grad", go[0].detach()))

    with torch.enable_grad():
        out      = model(inp)
        pred_idx = out.argmax(1).item()
        model.zero_grad()
        out[0, pred_idx].backward()

    fh.remove(); bh.remove()

    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0]

    w   = gc_obj.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * gc_obj.act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    rgb  = np.clip(inp.squeeze(0).cpu().numpy() * std + mean, 0, 1)
    rgb  = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
    hm   = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    hm   = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    overlay = (0.45 * hm + 0.55 * rgb).astype(np.uint8)

    return {
        "label":   CLASS_NAMES[pred_idx],
        "conf":    probs[pred_idx].item(),
        "auth_p":  probs[0].item(),
        "fake_p":  probs[1].item(),
        "rgb":     rgb,
        "overlay": overlay,
    }


def show(r):
    is_a  = r["label"] == "Authentic"
    color = "#10b981" if is_a else "#ef4444"
    vcls  = "verdict" if is_a else "verdict verdict-r"
    st.markdown(f'<div class="{vcls}" style="color:{color};">{r["label"].upper()}</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi"><div class="kpi-l">Confidence</div>'
                    f'<div class="kpi-v" style="color:{color};">{r["conf"]*100:.1f}%</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi"><div class="kpi-l">Authentic</div>'
                    f'<div class="kpi-v">{r["auth_p"]*100:.1f}%</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi kpi-r"><div class="kpi-l">Counterfeit</div>'
                    f'<div class="kpi-vr">{r["fake_p"]*100:.1f}%</div></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="hd">Grad-CAM</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a: st.image(r["rgb"],     caption="Original")
    with b: st.image(r["overlay"], caption="Activation Map")
    st.markdown(f'<div class="tip">Red/yellow = regions that drove the '
                f'<b style="color:{color};">{r["label"]}</b> decision. Blue = low influence.</div>',
                unsafe_allow_html=True)
    if "hist" not in st.session_state: st.session_state.hist = []
    st.session_state.hist.append({
        "label": r["label"], "conf": r["conf"],
        "time":  datetime.now().strftime("%H:%M")
    })


st.markdown("""
<div style="border-bottom:1px solid #0f0f0f;padding-bottom:.9rem;margin-bottom:1rem;">
  <div style="font-size:.58rem;color:#333;letter-spacing:.3em;text-transform:uppercase;">
    AI Pharmaceutical Authentication
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
              color:#fff;letter-spacing:-.04em;margin:.2rem 0 .3rem;">PharmaGuard</div>
  <div style="font-size:.62rem;color:#333;">
    96% accuracy · AUC 0.9947 · MobileNetV3 + Grad-CAM
  </div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["Upload File", "Webcam", "Mobile Camera"])

with t1:
    st.markdown('<div class="tip">Upload a tablet, capsule, blister pack, or '
                'packaging image. JPG / PNG / WEBP.</div>', unsafe_allow_html=True)
    f = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                         label_visibility="collapsed", key="f1")
    if f:
        img1 = Image.open(f).convert("RGB")
        st.image(img1, use_column_width=True)
    if st.button("Run Analysis", key="b1", disabled=not f):
        with st.spinner("Analysing..."):
            show(analyse(img1))

with t2:
    st.markdown('<div class="tip">Hold the medicine in front of your laptop camera. '
                'Works on localhost only (WebRTC requires HTTPS on remote).</div>',
                unsafe_allow_html=True)
    cam = st.camera_input("", label_visibility="collapsed", key="c2")
    if cam:
        img2 = Image.open(cam).convert("RGB")
    if st.button("Run Analysis", key="b2", disabled=not cam):
        with st.spinner("Analysing..."):
            show(analyse(img2))

with t3:
    st.markdown("""<div class="tip">
    <b style="color:#ccc;">Use your phone camera:</b><br>
    1. Connect phone to same Wi-Fi as this Mac<br>
    2. Your terminal shows: <span style="color:#10b981;">Network URL: http://192.168.x.x:8501</span><br>
    3. Open that URL on your phone<br>
    4. Tap "Browse files" below — choose "Take Photo" on iOS/Android<br><br>
    This uses the native OS camera picker (no HTTPS needed).
    </div>""", unsafe_allow_html=True)
    mob = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                           label_visibility="collapsed", key="m3")
    if mob:
        img3 = Image.open(mob).convert("RGB")
        st.image(img3, use_column_width=True)
    if st.button("Run Analysis", key="b3", disabled=not mob):
        with st.spinner("Analysing..."):
            show(analyse(img3))

if st.session_state.get("hist"):
    st.markdown('<div class="hd" style="margin-top:1.5rem;">History</div>',
                unsafe_allow_html=True)
    for h in reversed(st.session_state.hist[-5:]):
        c = "#10b981" if h["label"] == "Authentic" else "#ef4444"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'border-bottom:1px solid #080808;padding:.45rem 0;font-size:.72rem;">'
            f'<span style="color:{c};font-weight:600;">{h["label"]}</span>'
            f'<span>{h["conf"]*100:.1f}%</span>'
            f'<span style="color:#2a2a2a;">{h["time"]}</span></div>',
            unsafe_allow_html=True)

st.markdown('<div style="font-size:.58rem;color:#111;text-align:center;'
            'margin-top:2rem;padding-top:.6rem;border-top:1px solid #080808;">'
            'PharmaGuard — 96% Test Accuracy — AUC 0.9947</div>',
            unsafe_allow_html=True)