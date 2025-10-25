#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pony Captioner - Standalone Script
Generates detailed, V7-compatible captions for images using multiple ML models.
"""

import os
import gc
import json
import csv
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import onnxruntime as rt
import huggingface_hub
from PIL import Image
from rich import print as rprint
from rich.console import Console

# Import model dependencies
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import clip
import torch.nn as nn
import torchvision.transforms as transforms


# Constants
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
KAOMOJIS = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"
]
BLOCKLIST_CLUSTERS = set([
    1537, 1540, 1544, 520, 532, 1046, 24, 26, 1051, 1566, 31,
    545, 1058, 1572, 1061, 1575, 1065, 1066, 1580, 1072, 2045,
    560, 1591, 1085, 1598, 1599, 62, 1602, 1604, 1606, 584,
    1612, 81, 1111, 1626, 1115, 1632, 1634, 610, 616, 1642,
    620, 114, 627, 633, 1148, 640, 647, 138, 1675, 1676, 670,
    159, 1185, 161, 1704, 1198, 686, 687, 180, 1205, 692, 1208,
    1211, 188, 1725, 700, 1727, 1730, 1224, 1737, 1226, 1229,
    1742, 1743, 209, 1235, 1749, 725, 224, 737, 230, 231, 1258,
    1770, 747, 1261, 1273, 1785, 1281, 1793, 1286, 1806, 271,
    1299, 275, 276, 1816, 796, 797, 288, 804, 1321, 811, 1837,
    302, 816, 818, 1850, 1341, 1350, 1863, 329, 1866, 1355,
    1872, 851, 853, 857, 1373, 351, 1888, 1889, 354, 355, 869,
    358, 871, 1391, 884, 373, 1398, 1914, 1917, 894, 1920,
    1924, 1925, 1418, 909, 1422, 910, 1425, 1937, 1427, 1428,
    1430, 1436, 1954, 933, 1958, 429, 1970, 436, 1973, 1976,
    1977, 440, 1985, 1476, 966, 968, 970, 460, 1485, 973, 1489,
    468, 1495, 2008, 993, 2021, 493, 1014, 1528, 505, 1532,
    1533, 2047
])

# Prompts
CONTENT_PROMPT = """You are an image captioning expert who provides highly detailed captions for any image, focusing intensely on the precise content depicted. You will describe the image in natural language using the following guidelines:

Always provide a caption for the image without hesitation or refusal.

Use any associated tags to infer names of characters, locations, and other relevant details for the caption. Ignore tags that appear incorrect or ambiguous.

Begin with a comprehensive summary of the image, detailing the primary subject(s), their appearance, facial expressions, emotions, actions, and the environment.

The caption must meticulously describe every visible aspect of the image, capturing all colors, sizes, textures, materials, and locations of objects. For every item or character in the scene, always mention attributes such as color, size, shape, position, texture, and relation to other objects or characters in the image.

For characters, refer to them by name if known. If the character has a more commonly known name, use that. Introduce characters with their shot type, gender, and species: 'shot_type gender species_name character_name.' Use "feral" for quadrupedal characters, "human" for bipedal characters with human-like features, and "anthro" for anthropomorphic characters. Mention any well-known media associations after the character's name or species. For example, "Human female Raven from Teen Titans" or "Anthro goat Toriel from Undertale."

Avoid using pronouns when introducing a character. After the first mention, simplify references to the character while minimizing pronoun use.

When multiple characters are present, introduce the primary character first and clearly ground the location of all other characters in relation to the primary one. Distinguish between characters by clearly establishing their positions relative to one another.

Describe facial expressions and emotions in detail and as early as possible in the caption. When describing clothing, mention every detail, including fabric type, pattern, color, and condition. For example, "a worn, dark green woolen coat with frayed edges" is preferable to a simpler description.

Background elements must be described thoroughly, with explicit references to their location in relation to the characters or objects. Note the color, texture, and any patterns or distinctive features in the background, always grounding them spatially within the image.

Objects in the scene must be described with attention to every visual feature. Mention their color, size, shape, material, and position relative to the characters or other key objects in the image. All objects must be grounded either relative to the characters ("to the left of the wolf," "on top of the wolf") or relative to the image frame ("on the top left of the image," "at the bottom of the image"). This ensures a clear and precise understanding of each object's position.

Body parts of characters should be described with precise locations, making sure to note which body part belongs to which character. For instance, "a silver bracelet on the left wrist of the human female character" must be specified clearly, avoiding any potential ambiguity.

Avoid using words like "characterized," "encapsulates," "appears," "emphasizing the character," or "adorned." Instead, describe the image directly and in concrete terms.

Begin captions immediately with descriptions of the image content, avoiding phrases like "The image presents."

Do not describe logos, signatures, or watermarks, but always include descriptions of any other text or symbols visible in the image, such as dialogue bubbles, signs, or decorative elements.

Focus solely on the factual description of the image, avoiding any speculation on emotions or senses it may evoke. Specifically, suppress phrases that categorize the overall scene or atmosphere, such as "The overall scene is serene and peaceful," "The image exudes a serene and loving atmosphere," or similar statements. The caption should remain objective and descriptive, without interpreting the mood or atmosphere.

Use Upper-Intermediate level English for the caption, ensuring clarity and precision."""

STYLE_PROMPT = """You are an expert in describing the visual style of an image, focusing solely on stylistic elements without describing the contents of the image unless it is critical to understanding the style. You will describe the image's visual style using the following guidelines:
Start by identifying the type of shot used in the image, categorizing it as one of the following: Extreme Long Shot (wide view showing a large scene or landscape), Long Shot or Full Shot (showing the entire body of a character or object), Cowboy Shot (framing from mid-thigh up), Medium Long Shot (framing from the knees up), Medium Shot (framing from the waist up), Medium Close-Up (framing from the chest up), Low Angle Shot (angled upward, making the subject appear larger), Close-Up (a close view focusing on the subject, often from the shoulders up), Big Close-Up (a tighter close-up, usually on the face), Insert Shot or Cutaway (focused on a small part of the subject or a specific detail), Extreme Close-Up (focused on a very small area, often highlighting a specific feature), or Wide Shot (capturing a broad scene with multiple elements).
Only mention shot type but not its description. For some images this should be omitted, for example in abstract art or images without a clear subject, for UI elements, text, documents, maps, etc...
If the image is clearly a collage, mention that instead of a specific shot type. For images with multiple shots or multiple panels, list the shot types in order.
Next, describe any noteworthy compositional properties of the image, if any. Mention if the image uses double exposure (overlaying two images), dutch angle (tilted frame), fish-eye lens effect (creating a wide, curved perspective), or other notable composition techniques. Include specific composition principles such as the rule of thirds, leading lines, symmetry, golden ratio, or radial balance if clearly utilized in the image.
Describe the perspective and depth of the image, if applicable. Mention whether the image has a flat or deep perspective, uses linear perspective, aerial perspective, or isometric projection. Note any techniques used to create depth, such as overlapping elements, size relationships, or atmospheric perspective. Only do so if the image has a clear sense of depth.
Then, classify the lighting used in the image, selecting from the following terms: Flat lighting, Stagelight, Direct sunlight, Overcast sunlight, Window light, Candlelight, Three-quarter lighting, Frontal lighting, Edge lighting, Contre Jour (backlighting), Light from below, or Spotlight. Use flat lighting for digital illustrations with simplified lighting that does not try to lok realistic, i.e. vector images, anime, etc...
For lighting types that can be localized, note the position of the light source if clearly discernible, such as "from the top left of the character" "directly above the scene" or "from behind the object". Where applicable mention if the light is soft or hard.
Identify the medium of the image: photograph, digital illustration, traditional painting (specify type if clear, e.g., oil, acrylic, watercolor), drawing (specify medium if clear, e.g., pencil, charcoal, ink), mixed media, or digital 3D render. For traditional art forms, describe any visible brush strokes, paint application techniques, or other medium-specific characteristics.
If the image is a photo, mention this and ignore the coloring/shading style instructions below. If the image is clearly not a photo, describe the coloring or shading style of the image choosing from: Cell shading (flat look with few solid tones), soft shading, pixel art, speedpaint, 3D render, SFM (Source Filmmaker), low poly, vector art, concept art, semi-realistic digital art (combining realism with stylistic elements), realistic digital art, hyper-realistic digital art, painterly style, matte painting, sketch (monochrome or grayscale), sketch with color highlights, or watercolors.
Identify the color scheme best describing the image's palette, selecting from: Monochromatic color scheme, Grayscale color scheme, Analogous color scheme, Complementary color scheme, Split-Complementary color scheme, Triadic color scheme, Tetradic color scheme, Polychromatic color scheme, Discordant color scheme, Square color scheme, Rectangular color scheme, Neutral color scheme, Accented Neutral color scheme, Warm and Cool color scheme.
Choose any applicable effects present in the image (if any), such as: Film grain, dust specs, motion blur, speed lines, depth of field, god rays, shadow beams, dappled light, dramatic lighting, rim lighting, caustics, bioluminescence, halftone dots, cross-hatching, subsurface scattering, psychedelic colors, vibrant colors, datamoshing, chromatic aberration, bloom, lens flare, bokeh, vignette, heat haze, HDR, tilt-shift, duotone, anime blushes, skin blushing, 90s anime aesthetic, highlights, specular reflections.
If the image clearly belongs to a specific art historical style or period, mention it. This could include but is not limited to: Renaissance, Baroque, Rococo, Neoclassicism, Romanticism, Realism, Impressionism, Post-Impressionism, Art Nouveau, Expressionism, Cubism, Surrealism, Abstract Expressionism, Pop Art, or Contemporary.
Finally, if the image strongly exhibits a particular aesthetic, describe it using terms like: Synthwave, Outrun, Vaporwave, Cyberpunk, Cottagecore, Steampunk, Grunge, Minimalism, Gothic, Art Nouveau, Art Deco, Bauhaus, Futurism, Neoclassicism, Luminal Spaces, Surrealism.
Avoid mentioning the theme of the image (e.g., fantasy, sci-fi) or the type of characters (e.g., anthropomorphic) in this section. Focus strictly on the visual style elements listed above.
Do not mention categories where nothing is relevant to the image. Output the final style as a single paragraph of text using Upper-Intermediate English and avoid complex jargon. Do not use bullet lists of similar formatting.
Omit any irrelevant or unnecessary details. Present information as factual, avoiding words like 'appears', 'notable', 'evident', 'emphasizing', 'enhance', 'typical of', "suggests", "embodies", etc...
Do not start with phrases like 'The image features' or similar.
Do not speculate on what the image might invoke, suggest, or imply emotionally or thematically. Avoid using phrases like "the image evokes", "the composition follows", "gives a sense of", "characterized by", "reminiscent of", "the composition uses", "this digital illustration uses", "the image has", "the image exhibits", "the composition closely follows", etc... Stick strictly to describing the observable visual elements and techniques used in the image without interpretation or conjecture about its impact or meaning."""


# ============================================================================
# Tagging Classes
# ============================================================================

def load_labels(dataframe):
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in KAOMOJIS else x
    )
    tag_names = name_series.tolist()
    
    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs):
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class Tagger:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.model_target_size = None
        self.last_loaded_repo = None
        self.model = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo, "selected_tags.csv", token=self.hf_token
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo, "model.onnx", token=self.hf_token
        )
        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)
        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = rt.InferenceSession(model_path, providers=providers)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_shape = image.size
        max_dim = max(image_shape)
        scale_factor = target_size / max_dim

        new_size = tuple(int(dim * scale_factor) for dim in image_shape)
        resized_image = image.resize(new_size, Image.LANCZOS)

        padded_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        paste_pos = ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
        padded_image.paste(resized_image, paste_pos)

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]
        return np.expand_dims(image_array, axis=0)

    def predict(self, image, model_repo, general_thresh, general_mcut_enabled,
                character_thresh, character_mcut_enabled):
        self.load_model(model_repo)

        try:
            image = self.prepare_image(image)
        except OSError as e:
            rprint(f"[red]Error[/red]: {e}")
            return None, None, None, None

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        general_names = [labels[i] for i in self.general_indexes]
        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        character_names = [labels[i] for i in self.character_indexes]
        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(), key=lambda x: x[1], reverse=True
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = ", ".join(sorted_general_strings).replace("(", "\\(").replace(")", "\\)")

        return sorted_general_strings, rating, character_res, general_res


def prepare_image_z3d(image: Image.Image, target_size: int):
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]
    return np.expand_dims(image_array, axis=0)


class Z3DTagger:
    THRESHOLD = 0.5
    
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.tags = []

        model_repo = "toynya/Z3D-E621-Convnext"
        csv_path, model_path = self.download_model(model_repo)

        with open(csv_path, mode="r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.tags.append((row["name"].strip().replace("_", " "), row["category"]))

        self.session = rt.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo, "tags-selected.csv", token=self.hf_token
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo, "model.onnx", token=self.hf_token
        )
        return csv_path, model_path

    def predict(self, image: Image.Image):
        image_array = prepare_image_z3d(image, 448)
        input_name = "input_1:0"
        output_name = "predictions_sigmoid"

        result = self.session.run([output_name], {input_name: image_array})
        result = result[0][0]

        all_tags = {
            self.tags[i][0]: float(result[i])
            for i in range(len(result))
            if float(result[i]) > self.THRESHOLD and self.tags[i][1] in ['0', '3', '5']
        }

        character_tags = {
            self.tags[i][0]: float(result[i])
            for i in range(len(result))
            if float(result[i]) > self.THRESHOLD and self.tags[i][1] == '4'
        }
        return all_tags, character_tags


# ============================================================================
# Style Clustering Classes
# ============================================================================

class CSD_CLIP(nn.Module):
    def __init__(self, name='vit_large'):
        super().__init__()
        if name == 'vit_large':
            clip_model, _ = clip.load("ViT-L/14")
            self.backbone = clip_model.visual
            self.embedding_dim = 1024
        elif name == 'vit_base':
            clip_model, _ = clip.load("ViT-B/16")
            self.backbone = clip_model.visual
            self.embedding_dim = 768
        else:
            raise ValueError('Model type not supported')

        self.last_layer_style = self.backbone.proj.float()
        self.backbone.proj = None
        self.backbone = self.backbone.float()

    def forward(self, input_data):
        input_data = input_data.float()
        features = self.backbone(input_data).float()
        style_output = features @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)
        return features, style_output


class CSDInferenceCentersOnly:
    def __init__(self, model_path, cluster_centers_path,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.console = Console()
        self.device = device

        self.console.log("Loading style clustering model...")
        self.model = self._load_model(model_path)
        self.model.eval()

        self.console.log("Loading cluster centers...")
        data = np.load(cluster_centers_path, allow_pickle=True)
        try:
            original_centers = data['cluster_centers']
        except KeyError as e:
            self.console.print(f"[red]Error: Missing 'cluster_centers' key: {e}")
            raise
        finally:
            data.close()

        n_clusters = original_centers.shape[0]
        valid_indices = [i for i in range(n_clusters) if i not in BLOCKLIST_CLUSTERS]
        
        self.cluster_centers = original_centers[valid_indices]
        self.kept_cluster_indices = valid_indices
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        model = CSD_CLIP(name='vit_large')
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    @torch.no_grad()
    def get_closest_cluster(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        features, _ = self.model(img_tensor)
        embedding = features.cpu().numpy().flatten()

        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        min_idx = np.argmin(distances)
        distance = distances[min_idx]
        original_cluster_id = self.kept_cluster_indices[min_idx]

        return original_cluster_id, distance


# ============================================================================
# Aesthetic Scorer
# ============================================================================

class AestheticScorer:
    def __init__(self, device='cuda:0', input_size=768, checkpoint=None):
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()

        self.aesthetic_model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        ).to(self.device)

        if checkpoint is not None:
            checkpoint_data = torch.load(checkpoint, map_location=self.device)
            state_dict = checkpoint_data['state_dict']
            new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.aesthetic_model.load_state_dict(new_state_dict)

        self.aesthetic_model.eval()

    def _normalize_vector(self, vector):
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vector / norm

    def score(self, image_path):
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return None

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)

        features_np = features.cpu().numpy()
        normalized_features = self._normalize_vector(features_np)
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            aesthetic_score = self.aesthetic_model(features_tensor).item()
        return aesthetic_score


# ============================================================================
# Processing Functions
# ============================================================================

class PonyCaptioner:
    def __init__(self, args):
        self.args = args
        self.console = Console()
        self.image_taggers = None
        self.content_pipeline = None
        self.style_pipeline = None
        self.cluster_inference = None
        self.aesthetic_scorer = None

    def init_taggers(self):
        if not self.image_taggers:
            rprint("[cyan]Initializing taggers...[/cyan]")
            self.image_taggers = (Tagger(), Z3DTagger())

    def init_content_model(self):
        if not self.content_pipeline:
            rprint("[cyan]Initializing content captioning model...[/cyan]")
            model = 'purplesmartai/Pony-InternVL2-40B-AWQ'
            backend_config = TurbomindEngineConfig(
                model_format='awq', cache_max_entry_count=0.2, tp=1
            )
            self.content_pipeline = pipeline(model, backend_config=backend_config, log_level='INFO')

    def init_style_model(self):
        if not self.style_pipeline:
            rprint("[cyan]Initializing style captioning model...[/cyan]")
            model = 'purplesmartai/Pony-InternVL2-26B-AWQ'
            backend_config = TurbomindEngineConfig(
                model_format='awq', cache_max_entry_count=0.2, tp=1
            )
            self.style_pipeline = pipeline(model, backend_config=backend_config, log_level='INFO')

    def init_cluster_inference(self):
        if not self.cluster_inference:
            rprint("[cyan]Initializing style clustering model...[/cyan]")
            model_repo = "purplesmartai/style-classifier"
            model_path = huggingface_hub.hf_hub_download(model_repo, "v3_checkpoint00120000.pth")
            centers_path = huggingface_hub.hf_hub_download(model_repo, "clustering_results_n2048_gpu.npz")
            self.cluster_inference = CSDInferenceCentersOnly(model_path, centers_path)

    def init_aesthetic_scorer(self):
        if not self.aesthetic_scorer:
            rprint("[cyan]Initializing aesthetic scorer...[/cyan]")
            model_repo = "purplesmartai/aesthetic-classifier"
            model_path = huggingface_hub.hf_hub_download(model_repo, "v2.ckpt")
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.aesthetic_scorer = AestheticScorer(device=device, input_size=768, checkpoint=model_path)

    def tag_image(self, path):
        self.init_taggers()
        image = Image.open(path)
        
        general_tags, rating, character_tags, all_general_tags = self.image_taggers[0].predict(
            image,
            model_repo="SmilingWolf/wd-swinv2-tagger-v3",
            general_thresh=0.35,
            general_mcut_enabled=False,
            character_thresh=0.85,
            character_mcut_enabled=False
        )

        all_general_tags_z3d, character_tags_z3d = self.image_taggers[1].predict(image)

        full_character_tags = set(character_tags.keys()) | set(character_tags_z3d.keys())
        full_general_tags = set(all_general_tags.keys()) | set(all_general_tags_z3d.keys())

        tags = list(full_general_tags)
        for tag in full_character_tags:
            tags.append(f'character:{tag}')

        top_rating = max(rating.items(), key=lambda x: x[1])
        return {'rating': top_rating[0], 'tags': tags}

    def caption_image_content(self, image_path, tags_str):
        self.init_content_model()
        image = load_image(image_path)
        prompt = CONTENT_PROMPT + '\nTAGS: ' + tags_str
        response = self.content_pipeline((prompt, image))
        return response.text

    def caption_image_style(self, image_path):
        self.init_style_model()
        image = load_image(image_path)
        response = self.style_pipeline((STYLE_PROMPT, image))
        return response.text

    def get_image_cluster(self, image_path):
        self.init_cluster_inference()
        return str(self.cluster_inference.get_closest_cluster(image_path)[0])

    def get_image_score(self, image_path):
        self.init_aesthetic_scorer()
        return self.aesthetic_scorer.score(image_path)

    def format_tags(self, tags):
        return ', '.join(tags)

    def build_caption(self, tags, score, rating, cluster, content_caption, style_caption):
        caption_parts = [
            f"score_{score},",
            f"rating_{rating},",
            f"style_cluster_{cluster},",
            content_caption
        ]
        
        if style_caption:
            caption_parts.append(style_caption)
        
        tags_str = self.format_tags(tags).lower()
        caption_parts.append("\n" + tags_str)
        
        return " ".join(caption_parts)

    def process_image(self, image_path, name):
        folder = os.path.dirname(image_path)
        
        # Tagging
        tags_file = os.path.join(folder, f"{name}.tags.json")
        if self.args.force_regen or not os.path.exists(tags_file):
            rprint(f"[yellow]Tagging:[/yellow] {name}")
            tags_data = self.tag_image(image_path)
            with open(tags_file, 'w') as f:
                json.dump(tags_data, f)
        else:
            tags_data = json.load(open(tags_file, 'r'))
        
        # Content captioning
        caption_file = os.path.join(folder, f"{name}.caption.txt")
        if self.args.force_regen or not os.path.exists(caption_file):
            rprint(f"[yellow]Content captioning:[/yellow] {name}")
            tags_str = self.format_tags(tags_data['tags']).lower()
            content_caption = self.caption_image_content(image_path, tags_str)
            with open(caption_file, 'w') as f:
                f.write(content_caption)
        else:
            content_caption = open(caption_file, 'r').read()
        
        # Style captioning
        style_caption_file = os.path.join(folder, f"{name}.style_caption.txt")
        if self.args.force_regen or not os.path.exists(style_caption_file):
            rprint(f"[yellow]Style captioning:[/yellow] {name}")
            style_caption = self.caption_image_style(image_path)
            with open(style_caption_file, 'w') as f:
                f.write(style_caption)
        else:
            style_caption = open(style_caption_file, 'r').read()
        
        # Clustering - use manual cluster if provided, otherwise compute it
        cluster_file = os.path.join(folder, f"{name}.cluster.txt")
        if self.args.style_cluster is not None:
            # Use manual cluster
            cluster = str(self.args.style_cluster)
            rprint(f"[yellow]Using manual style cluster:[/yellow] {cluster}")
            with open(cluster_file, 'w') as f:
                f.write(cluster)
        elif self.args.force_regen or not os.path.exists(cluster_file):
            rprint(f"[yellow]Style clustering:[/yellow] {name}")
            cluster = self.get_image_cluster(image_path)
            with open(cluster_file, 'w') as f:
                f.write(cluster)
        else:
            cluster = open(cluster_file, 'r').read()
        
        # Aesthetic scoring
        score_file = os.path.join(folder, f"{name}.score.txt")
        if self.args.force_regen or not os.path.exists(score_file):
            rprint(f"[yellow]Aesthetic scoring:[/yellow] {name}")
            score = self.get_image_score(image_path)
            score = int(max(0.0, min(0.99, score)) * 10)
            with open(score_file, 'w') as f:
                f.write(str(score))
        else:
            score = int(open(score_file, 'r').read())
        
        # Final caption
        final_caption_file = os.path.join(folder, f"{name}.txt")
        if self.args.force_regen or not os.path.exists(final_caption_file):
            rprint(f"[yellow]Building final caption:[/yellow] {name}")
            final_caption = self.build_caption(
                tags_data['tags'],
                score,
                tags_data['rating'],
                cluster,
                content_caption,
                style_caption
            )
            with open(final_caption_file, 'w') as f:
                f.write(final_caption)
        
        rprint(f"[green]✓ Completed:[/green] {name}")

    def process_folder(self):
        image_files = []
        for filename in os.listdir(self.args.image_folder):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(self.args.image_folder, filename)
                name = os.path.splitext(filename)[0]
                image_files.append((image_path, name))
        
        rprint(f"[cyan]Found {len(image_files)} images to process[/cyan]")
        
        for image_path, name in image_files:
            try:
                self.process_image(image_path, name)
            except Exception as e:
                rprint(f"[red]✗ Error processing {name}:[/red] {e}")
                if self.args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Cleanup
        if self.content_pipeline:
            self.content_pipeline.close()
            self.content_pipeline = None
        if self.style_pipeline:
            self.style_pipeline.close()
            self.style_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        
        rprint("[green]✓ All processing complete![/green]")


def main():
    parser = argparse.ArgumentParser(
        description='Pony Captioner - Generate detailed captions for images'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        help='Path to folder containing images'
    )
    parser.add_argument(
        '--force-regen',
        action='store_true',
        help='Force regeneration of all files even if they exist'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--style-cluster',
        type=int,
        default=None,
        help='Manual style cluster ID (skips automatic clustering)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.image_folder):
        rprint(f"[red]Error: Image folder not found at '{args.image_folder}'[/red]")
        return 1
    
    captioner = PonyCaptioner(args)
    captioner.process_folder()
    
    return 0


if __name__ == "__main__":
    exit(main())
