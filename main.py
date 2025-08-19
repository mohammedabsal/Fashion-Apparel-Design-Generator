""" Fully Functional Web App — Fashion Apparel Designer Generator (cGAN-ready)

Run locally:

1. Create a venv (optional) and install deps pip install -r requirements.txt

or minimal: pip install streamlit torch torchvision pillow numpy scikit-image

2. Start the app streamlit run app.py

3. (Optional) Plug in your trained cGAN

Save your PyTorch generator weights to ./weights/generator.pth

The file must load with the Generator class defined below (nz=128, nclass=6 by default).

When present, the app automatically switches from the procedural renderer to the GAN generator.

Notes

Ships with an integrated AI Design Agent that parses natural language prompts into category + attributes and suggests color palettes. It also offers one-click latent evolution.

Out-of-the-box it produces believable apparel mockups using a fast procedural renderer (no training needed). When you drop in GAN weights, outputs switch to learned samples.

File layout

app.py  (this file)
requirements.txt (auto-written on first run if missing)
weights/generator.pth (optional)
"""
from streamlit_lottie import st_lottie
import requests
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
from ast import pattern
import os
import io
import math
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Optional torch import (the app works even if torch isn't installed — it will stay in procedural mode)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import streamlit as st

APP_VERSION = "1.0.0"

# -------------- Utility: ensure requirements.txt exists for convenience --------------
REQ_TXT = "requirements.txt"
REQ_CONTENT = """
streamlit>=1.30
numpy
Pillow
scikit-image

# Optional for GAN mode:
torch
torchvision
"""
if not os.path.exists(REQ_TXT):
    try:
        with open(REQ_TXT, "w", encoding="utf-8") as f:
            f.write(REQ_CONTENT)
    except Exception:
        pass

# -------------- Categories & attribute lexicons --------------

CATEGORIES = ["shirt", "tshirt", "dress", "jacket", "skirt", "pants"]
PATTERNS = ["solid", "floral", "stripes", "polka", "geom", "abstract"]
STYLES = ["boho", "casual", "formal", "street", "vintage", "sport"]
LENGTHS = ["crop", "mini", "midi", "maxi", "knee", "ankle"]
NECKLINES = ["vneck", "round", "collared", "turtleneck", "square"]
SLEEVES = ["sleeveless", "short", "threequarter", "long"]

# Default palette bank

PALETTES = {
    "boho": [(178, 102, 70), (226, 185, 141), (48, 92, 75), (160, 61, 67)],
    "casual": [(52, 73, 94), (46, 134, 193), (244, 246, 247), (231, 76, 60)],
    "formal": [(33, 47, 60), (127, 140, 141), (236, 240, 241), (22, 160, 133)],
    "street": [(20, 20, 20), (250, 197, 28), (231, 76, 60), (52, 152, 219)],
    "vintage": [(112, 66, 20), (196, 154, 108), (116, 140, 171), (66, 90, 73)],
    "sport": [(0, 0, 0), (230, 126, 34), (41, 128, 185), (236, 240, 241)],
}

# -------------- AI Design Agent: prompt parsing --------------

@dataclass
class DesignSpec:
    category: str
    pattern: str
    style: str
    length: str
    neckline: str
    sleeve: str
    palette: List[Tuple[int, int, int]]
    seed: int

KEYWORD_MAP = {
    # styles
    "boho": ("style", "boho"),
    "street": ("style", "street"),
    "formal": ("style", "formal"),
    "vintage": ("style", "vintage"),
    "sport": ("style", "sport"),
    "casual": ("style", "casual"),
    # patterns
    "floral": ("pattern", "floral"),
    "flowers": ("pattern", "floral"),
    "polka": ("pattern", "polka"),
    "dots": ("pattern", "polka"),
    "striped": ("pattern", "stripes"),
    "stripes": ("pattern", "stripes"),
    "geometric": ("pattern", "geom"),
    "abstract": ("pattern", "abstract"),
    # lengths
    "midi": ("length", "midi"),
    "mini": ("length", "mini"),
    "maxi": ("length", "maxi"),
    "knee": ("length", "knee"),
    "ankle": ("length", "ankle"),
    "crop": ("length", "crop"),
    # necklines
    "v neck": ("neckline", "vneck"),
    "v-neck": ("neckline", "vneck"),
    "vneck": ("neckline", "vneck"),
    "collar": ("neckline", "collared"),
    "turtleneck": ("neckline", "turtleneck"),
    "round neck": ("neckline", "round"),
    # sleeves
    "sleeveless": ("sleeve", "sleeveless"),
    "short sleeve": ("sleeve", "short"),
    "short sleeves": ("sleeve", "short"),
    "3/4": ("sleeve", "threequarter"),
    "long sleeve": ("sleeve", "long"),
    "long sleeves": ("sleeve", "long"),
}

def parse_prompt_to_spec(prompt: str, seed: int, default_category: str) -> DesignSpec:
    p = prompt.lower()
    fields = {
        "category": default_category,
        "pattern": "solid",
        "style": "casual",
        "length": "midi",
        "neckline": "round",
        "sleeve": "short",
    }
    # category hints in text
    for cat in CATEGORIES:
        if cat in p:
            fields["category"] = cat
            break
    # keyword mapping
    for k, (field, val) in KEYWORD_MAP.items():
        if k in p:
            fields[field] = val
    # palette extraction by style
    palette = PALETTES.get(fields["style"], PALETTES["casual"])[:]
    # color hints
    color_words = {
        "red": (231, 76, 60), "blue": (52, 152, 219), "green": (39, 174, 96), "yellow": (241, 196, 15),
        "orange": (230, 126, 34), "purple": (155, 89, 182), "pink": (243, 156, 18), "black": (0,0,0),
        "white": (255,255,255), "gray": (127,140,141), "teal": (22,160,133), "navy": (44,62,80)
    }
    custom_colors = []
    for w, rgb in color_words.items():
        if w in p:
            custom_colors.append(rgb)
    if custom_colors:
        palette = (custom_colors + palette)[:4]
    return DesignSpec(
        category=fields["category"],
        pattern=fields["pattern"],
        style=fields["style"],
        length=fields["length"],
        neckline=fields["neckline"],
        sleeve=fields["sleeve"],
        palette=palette,
        seed=seed,
    )

# -------------- Procedural renderer (fast, no-ML fallback) --------------

def perlin_noise(h, w, scale=8.0, seed=0):
    rng = np.random.default_rng(seed)
    grid_y = int(max(1, h/scale))
    grid_x = int(max(1, w/scale))
    gradients = rng.standard_normal((grid_y+1, grid_x+1, 2))
    gradients /= np.linalg.norm(gradients, axis=2, keepdims=True) + 1e-9
    ys = np.linspace(0, grid_y, h, endpoint=False)
    xs = np.linspace(0, grid_x, w, endpoint=False)
    yi = ys.astype(int)
    xi = xs.astype(int)
    yf = ys - yi
    xf = xs - xi
    def fade(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    u = fade(xf)[:, None]
    v = fade(yf)[None, :]
    out = np.zeros((h, w))
    for dy in (0,1):
        for dx in (0,1):
            g = gradients[yi+dy, xi+dx]
            d = np.stack([xf-dx, yf-dy], axis=-1)
            dot = g[..., 0] * d[..., 0] + g[..., 1] * d[..., 1]
            wx = (u if dx==1 else (1-u))
            wy = (v if dy==1 else (1-v))
            out += (wx*wy).T * dot
    out = (out - out.min())/(out.max()-out.min()+1e-9)
    return out

# ...existing code for make_pattern, garment_mask, Generator, evolve_latent, generate_image, and Streamlit UI...

def make_pattern(h, w, pattern: str, palette: List[Tuple[int,int,int]], seed: int):
    rng = np.random.default_rng(seed)
    base = perlin_noise(h, w, scale=12.0, seed=seed)
    img = np.zeros((h,w,3), dtype=np.uint8)
    # map noise to palette
    bins = np.digitize(base, np.linspace(0,1,len(palette)+1)[1:-1])
    colors = np.array(palette, dtype=np.uint8)
    img = colors[bins]
    if pattern == "solid":
        c = colors[rng.integers(0, len(colors))]
        img[:,:] = c
    elif pattern == "stripes":
        stripe_w = rng.integers(6, 20)
        for x in range(w):
            c = colors[(x//stripe_w) % len(colors)]
            img[:, x] = c
    elif pattern == "polka":
        img[:,:] = colors[0]
        dots = rng.integers(8, 18)
        for _ in range(250):
            y = rng.integers(0, h)
            x = rng.integers(0, w)
            r = rng.integers(dots-4, dots+6)
            c = tuple(colors[rng.integers(1, len(colors))])
            ImageDraw.Draw(Image.fromarray(img)).ellipse((x-r,y-r,x+r,y+r), fill=c)
            img = np.array(Image.fromarray(img))
    elif pattern == "floral":
        img[:,:] = colors[0]
        canvas = Image.fromarray(img)
        draw = ImageDraw.Draw(canvas)
        for _ in range(120):
            y = rng.integers(0, h)
            x = rng.integers(0, w)
            r = rng.integers(8, 16)
            petal_c = tuple(colors[rng.integers(1, len(colors))])
            for k in range(6):
                ang = 2*math.pi*k/6
                dx = int(r*math.cos(ang))
                dy = int(r*math.sin(ang))
                draw.ellipse((x-dx-6,y-dy-6,x-dx+6,y-dy+6), fill=petal_c)
            draw.ellipse((x-5,y-5,x+5,y+5), fill=tuple(colors[-1]))
        img = np.array(canvas.filter(ImageFilter.GaussianBlur(0.6)))
    elif pattern == "geom":
        img[:,:] = colors[0]
        canvas = Image.fromarray(img)
        draw = ImageDraw.Draw(canvas)
        for _ in range(220):
            y = rng.integers(0, h)
            x = rng.integers(0, w)
            s = rng.integers(8, 20)
            c = tuple(colors[rng.integers(1, len(colors))])
            if rng.random() < 0.5:
                draw.rectangle((x, y, x+s, y+s), fill=c)
            else:
                draw.polygon([(x,y),(x+s,y),(x+s//2,y+s)], fill=c)
        img = np.array(canvas)
    else:  # abstract
        n_layers = 4
        canvas = Image.fromarray(img)
        for i in range(n_layers):
            layer = (perlin_noise(h, w, scale=8+4*i, seed=seed+i)*255).astype(np.uint8)
            tint = np.array(palette[i%len(palette)], dtype=np.float32)
            colored = np.stack([layer, layer, layer], axis=-1).astype(np.float32)
            colored = (colored * 0.5 + tint * 0.5).clip(0,255).astype(np.uint8)
            canvas = Image.blend(canvas, Image.fromarray(colored), alpha=0.5)
        img = np.array(canvas)
    return Image.fromarray(img)

# -------------- Silhouette masks for garments --------------

def garment_mask(category: str, size=(512,512), length="midi", sleeve="short", neckline="round") -> Image.Image:
    w, h = size
    img = Image.new("L", size, 0)
    d = ImageDraw.Draw(img)
    cx, cy = w//2, h//2

    if category in ("shirt", "tshirt", "jacket"):
        body_w = int(w*0.55)
        body_h = int(h*0.55)
        top = int(h*0.18)
        left = cx - body_w//2
        d.rounded_rectangle((left, top, left+body_w, top+body_h), radius=40, fill=255)
        # sleeves
        if sleeve in ("short", "sleeveless"):
            s_h = int(h*0.18)
        elif sleeve == "threequarter":
            s_h = int(h*0.28)
        else:
            s_h = int(h*0.38)
        s_w = int(w*0.22)
        d.rectangle((left - s_w, top+40, left, top+40+s_h), fill=255)
        d.rectangle((left+body_w, top+40, left+body_w+s_w, top+40+s_h), fill=255)
        # neckline
        if neckline == "vneck":
            d.polygon([(cx-40, top+10), (cx+40, top+10), (cx, top+60)], fill=0)
        elif neckline == "collared":
            d.polygon([(cx-50, top+10),(cx, top+45),(cx-10, top+10)], fill=0)
            d.polygon([(cx+50, top+10),(cx, top+45),(cx+10, top+10)], fill=0)
        elif neckline == "turtleneck":
            d.rectangle((cx-60, top, cx+60, top+35), fill=255)
        # jacket opening
        if category == "jacket":
            d.rectangle((cx-8, top+40, cx+8, top+body_h), fill=0)
    elif category == "dress":
        top = int(h*0.12)
        bod_h = int(h*0.25)
        skirt_top = top + bod_h
        # bodice
        d.rounded_rectangle((cx-int(w*0.18), top, cx+int(w*0.18), skirt_top), radius=40, fill=255)
        # skirt based on length
        if length == "mini":
            skirt_bottom = int(h*0.55)
        elif length == "midi":
            skirt_bottom = int(h*0.75)
        elif length == "maxi":
            skirt_bottom = int(h*0.92)
        else:
            skirt_bottom = int(h*0.7)
        d.polygon([(cx-int(w*0.4), skirt_top), (cx+int(w*0.4), skirt_top), (cx+int(w*0.25), skirt_bottom), (cx-int(w*0.25), skirt_bottom)], fill=255)
        # neckline
        if neckline == "vneck":
            d.polygon([(cx-40, top+5), (cx+40, top+5), (cx, top+60)], fill=0)
        elif neckline == "square":
            d.rectangle((cx-55, top+5, cx+55, top+40), fill=0)
    elif category == "skirt":
        top = int(h*0.35)
        if length == "mini":
            bottom = int(h*0.55)
        elif length == "midi":
            bottom = int(h*0.75)
        else:
            bottom = int(h*0.92)
        d.polygon([(cx-int(w*0.45), top), (cx+int(w*0.45), top), (cx+int(w*0.25), bottom), (cx-int(w*0.25), bottom)], fill=255)
    elif category == "pants":
        top = int(h*0.28)
        bottom = int(h*0.92)
        leg_w = int(w*0.22)
        gap = int(w*0.06)
        d.rectangle((cx-gap-leg_w, top, cx-gap, bottom), fill=255)
        d.rectangle((cx+gap, top, cx+gap+leg_w, bottom), fill=255)
        # waist
        d.rectangle((cx-int(w*0.35), top-20, cx+int(w*0.35), top+20), fill=255)
    else:
        d.ellipse((cx-100, cy-140, cx+100, cy+140), fill=255)

    return img

# -------------- GAN model definitions (DCGAN-like) --------------

class Generator(nn.Module):
    def __init__(self, nz=128, nclass=len(CATEGORIES)):
        super().__init__()
        self.nz = nz
        self.nclass = nclass
        ngf = 64
        self.embed = nn.Embedding(nclass, nclass)
        self.fc = nn.Sequential(
            nn.Linear(nz + nclass, ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, y):
        # y is class idx tensor [B]
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.nclass).float()
        h = torch.cat([z, y_onehot], dim=1)
        h = self.fc(h)
        h = h.view(h.size(0), -1, 4, 4)
        x = self.deconv(h)
        return x

# -------------- Latent utilities --------------

def evolve_latent(z: np.ndarray, power: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    step = rng.standard_normal(z.shape).astype(np.float32)
    step = step / (np.linalg.norm(step) + 1e-8)
    return (z + power * step).astype(np.float32)

# -------------- Core generation path --------------

def generate_image(
    spec: DesignSpec,
    use_gan: bool,
    gan_device: str,
    z: Optional[np.ndarray],
    y_idx: int,
    out_size=(512,512)
) -> Image.Image:
    random.seed(spec.seed)
    np.random.seed(spec.seed)

    mask = garment_mask(spec.category, size=out_size, length=spec.length, sleeve=spec.sleeve, neckline=spec.neckline)
    pattern_img = make_pattern(out_size[1], out_size[0], spec.pattern, spec.palette, seed=spec.seed)
    canvas = Image.new("RGBA", out_size, (245,245,245,255))

    if use_gan and TORCH_AVAILABLE:
        try:
            device = torch.device(gan_device)
            gen = Generator(nz=128, nclass=len(CATEGORIES))
            weights_path = os.path.join("weights", "generator.pth")
            gen.load_state_dict(torch.load(weights_path, map_location=device))
            gen.to(device).eval()
            if z is None:
                z = np.random.randn(1, 128).astype(np.float32)
            zt = torch.from_numpy(z).to(device)
            yt = torch.tensor([y_idx], dtype=torch.long, device=device)
            with torch.no_grad():
                out = gen(zt, yt).cpu().numpy()[0]
            out = ((out.transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
            tex = Image.fromarray(out).resize(out_size, Image.BICUBIC)
        except Exception as e:
            # fallback if weights missing or incompatible
            tex = pattern_img
    else:
        tex = pattern_img

    # apply mask
    tex_rgba = tex.convert("RGBA")
    m = mask.filter(ImageFilter.GaussianBlur(0.5))
    tex_rgba.putalpha(m)

    # subtle mannequin hint
    mannequin = Image.new("RGBA", out_size, (0,0,0,0))
    md = ImageDraw.Draw(mannequin)
    md.ellipse((out_size[0]//2-35, int(out_size[1]*0.05), out_size[0]//2+35, int(out_size[1]*0.17)), fill=(230,230,230,140))

    # compose
    canvas.alpha_composite(mannequin)
    canvas.alpha_composite(tex_rgba)

    # shadow
    shadow = m.filter(ImageFilter.GaussianBlur(6))
    sh = Image.new("RGBA", out_size, (0,0,0,0))
    sh_np = np.array(sh)
    sh_np[...,3] = (np.array(shadow) * 0.3).astype(np.uint8)
    sh = Image.fromarray(sh_np, mode="RGBA")
    canvas = Image.alpha_composite(Image.new("RGBA", out_size, (250,250,250,255)), sh)
    canvas = Image.alpha_composite(canvas, tex_rgba)

    return canvas.convert("RGB")
# -------------- Streamlit UI --------------

st.set_page_config(page_title="cGAN Fashion Designer", page_icon="👗", layout="wide")

with st.sidebar:
    st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"), height=120, key="fashion_anim")
    st.markdown("<h2 style='color:#6C3483;'>👗 Fashion Designer</h2>", unsafe_allow_html=True)
    st.caption(f"v{APP_VERSION} · Works offline · Drop GAN weights for ML generation")
    st.markdown("---")
    st.subheader("Generation Mode")
    gan_device = st.selectbox("GAN device", ["cpu", "cuda"], index=0)
    weights_present = os.path.exists(os.path.join("weights", "generator.pth"))
    use_gan = st.toggle("Use GAN (if weights available)", value=weights_present)
    if use_gan and not weights_present:
        st.info("No generator weights found at ./weights/generator.pth — using procedural renderer.")
    st.markdown("---")
    st.subheader("Category & Attributes")
    default_cat = st.selectbox("Category", CATEGORIES, index=0)
    pattern = st.selectbox("Pattern", PATTERNS, index=0)
    style = st.selectbox("Style", STYLES, index=1)
    length = st.selectbox("Length", LENGTHS, index=2)
    neckline = st.selectbox("Neckline", NECKLINES, index=1)
    sleeve = st.selectbox("Sleeve", SLEEVES, index=1)
    st.markdown("---")
    st.subheader("Palette")
    base_palette = PALETTES.get(style, PALETTES["casual"])[:]
    custom_palette = []
    for i, c in enumerate(base_palette):
        color = st.color_picker(f"Color {i+1}", value="#%02x%02x%02x"%c)
        custom_palette.append(tuple(int(color[j:j+2],16) for j in (1,3,5)))
    st.markdown("---")
    st.subheader("Latent Controls")
    seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
    evolve_power = st.slider("Latent evolve power", 0.0, 3.0, 0.7, 0.1)
    st.markdown("---")
    st.subheader("Example Prompts")
    st.markdown("""
    <span style='font-size:15px'>
    • boho floral midi dress in teal & mustard, v-neck, sleeveless<br>
    • casual striped t-shirt, round neck, short sleeves, blue and white<br>
    • formal black maxi skirt, high waist, ankle length<br>
    • sporty orange jacket, zip-up, long sleeves<br>
    • elegant formal evening dress, maxi length, off-shoulder, navy blue and gold<br>
    • street style cropped jacket, geometric pattern, long sleeves, black and yellow<br>
    • vintage midi skirt, floral pattern, high waist, pastel colors<br>
    • sporty t-shirt, stripes, round neck, short sleeves, red and white<br>
    • casual boho blouse, polka dots, v-neck, threequarter sleeves, teal and coral<br>
    • summer mini dress, sleeveless, abstract pattern, pink and orange<br>
    • winter coat, solid color, turtleneck, long sleeves, gray and burgundy<br>
    • business formal pants, ankle length, solid navy, high waist<br>
    • party dress, metallic shimmer, square neckline, sleeveless, silver<br>
    • retro jumpsuit, bold stripes, collared, long sleeves, blue and green<br>
    </span>
    """, unsafe_allow_html=True)

st.title("Fashion Apparel Designer — cGAN (with AI Design Agent)")
st.markdown("<hr>", unsafe_allow_html=True)

left, right = st.columns([2,1])

with right:
    st.subheader("AI Design Agent")
    prompt = st.text_area("Describe your vibe", value="boho floral midi dress in teal & mustard, v-neck, sleeveless")
    if st.button("✨ Parse Prompt"):
        with st.spinner("Parsing prompt and generating spec..."):
            spec = parse_prompt_to_spec(prompt, seed=seed, default_category=default_cat)
            st.session_state["spec"] = spec
            st.success(f"Parsed → {spec.category}, {spec.style}, {spec.pattern}, {spec.length}, {spec.neckline}, {spec.sleeve}")
            st.session_state["palette"] = spec.palette
    spec: DesignSpec = st.session_state.get("spec") or DesignSpec(
        category=default_cat,
        pattern=pattern,
        style=style,
        length=length,
        neckline=neckline,
        sleeve=sleeve,
        palette=custom_palette,
        seed=seed
    )
    if "palette" in st.session_state and st.session_state["palette"]:
        spec.palette = st.session_state["palette"]
    else:
        spec.palette = custom_palette
    if "z" not in st.session_state:
        st.session_state["z"] = np.random.randn(1,128).astype(np.float32)
    if st.button("🔁 Evolve Latent"):
        with st.spinner("Evolving latent vector..."):
            st.session_state["z"] = evolve_latent(st.session_state["z"], power=evolve_power, seed=seed)

with left:
    st.subheader("Design Canvas")
    y_idx = CATEGORIES.index(spec.category)
    with st.spinner("Generating design..."):
        img = generate_image(spec, use_gan=use_gan, gan_device=gan_device, z=st.session_state.get("z"), y_idx=y_idx, out_size=(768,768))
    st.image(img, caption=f"{spec.style} {spec.category} · {spec.pattern} · {spec.length} · {spec.neckline} · {spec.sleeve}", use_container_width=True)
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("🎨 Shuffle Palette"):
            random.shuffle(spec.palette)
            st.session_state["palette"] = spec.palette
            st.rerun()
    with colB:
        if st.button("🎲 New Seed"):
            st.session_state["z"] = np.random.randn(1,128).astype(np.float32)
            st.rerun()
    with colC:
        if st.button("💾 Download PNG"):
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            st.download_button(
                label="Save image", data=buf.getvalue(), file_name=f"design_{spec.category}_{spec.style}.png", mime="image/png"
            )

st.markdown("<hr>", unsafe_allow_html=True)

# --- New: Add a "How to use" expandable section at the bottom ---
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Choose garment attributes** in the sidebar (category, pattern, style, etc).
    2. **Pick or customize your palette** using the color pickers.
    3. **Describe your vibe** in the AI Design Agent box and click "✨ Parse Prompt" for smart suggestions.
    4. **Evolve latent** for new GAN-based variations.
    5. **Shuffle palette** or **generate new seed** for more variety.
    6. **Download your design** as PNG.
    """)


# '''
# How it works
#
# Procedural Mode: We generate a garment silhouette, then fill it with an attribute-aware pattern (floral/stripes/polka/geom/abstract) colored by the AI agent's palette.
#
# GAN Mode: If you provide weights/generator.pth, the app switches to a DCGAN-like conditional generator. Conditioning is the category; you can extend it to include pattern/style by expanding the embedding & one-hot.
#
# Training your cGAN (sketch)
#
# # Pseudocode snippet (train.py)
# G = Generator(nz=128, nclass=len(CATEGORIES)).to(device)
# # Define Discriminator D with class-conditional input
# # Train on your fashion dataset with labels (category)
# # Save: torch.save(G.state_dict(), 'weights/generator.pth')
#
# Tips
#
# Want text-driven refinement? Connect CLIP to score matches between the prompt and generated image, and ascend z.
#
# Want multi-attribute conditioning? Concatenate one-hots for pattern/style/length to the Generator input.
#
# Want better silhouettes? Replace garment_mask with vector SVG patterns from real tech-packs.
# '''