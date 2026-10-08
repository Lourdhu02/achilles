# Cheat sheets

High-density, printable 2-column reference sheets that condense foundational math, system architectures, runnable PyTorch implementations, napkin scaling, debugging playbooks, and interview cards for revision.

| File | Specs | Description |
|---|---|---|
| [llm-genai-cheatsheet.pdf](llm-genai-cheatsheet.pdf) | 24 pages, 2 columns | **LLM & Generative AI Engineer**: Math, GPU hardware, pretraining, post-training (RLHF/GRPO), inference serving, RAG, agents, from-scratch code, debugging, formulas, and papers. |
| [vision-cheatsheet.pdf](vision-cheatsheet.pdf) | 24 pages, 2 columns | **Vision & Multimodal AI Engineer**: Classical geometry, CNNs, ViTs, Detection (YOLOv1--v11, DETR), SAM 1/2, SSL (MAE/DINOv2), CLIP/SigLIP, VLMs (LLaVA/Qwen2-VL), Diffusion (DDPM/EDM), DiT, Flow Matching (Flux), Video, 3DGS, Edge NPUs, and training loops. |
| [life-of-a-token.pdf](life-of-a-token.pdf) | 5 pages, 2 columns | Narrative recap: The life of a token told as one continuous story across 9 stages. |

---

## Build Instructions

Each document is compiled using TeX Live / TinyTeX (`pdflatex`) with `mathpazo`, `tcolorbox`, `listings`, `pgfplots`, `microtype`, and `booktabs`. Run `pdflatex` twice to populate the Table of Contents and cross-references.

### 1. Build Vision Cheatsheet (24 Pages)
```bash
cd cheatsheet/vision/src
pdflatex main.tex
pdflatex main.tex
```

### 2. Build LLM & GenAI Cheatsheet (24 Pages)
```bash
cd cheatsheet/src
pdflatex main.tex
pdflatex main.tex
```

### 3. Build Story PDF
```bash
cd cheatsheet/story
pdflatex life-of-a-token.tex
pdflatex life-of-a-token.tex
```

---

## Vision Cheatsheet Section Directory (`vision/src/sec/`)

- `00-front.tex` --- Title, study method, 12 numbers everyone asks, Table of Contents
- `01-geometry-filtering.tex` --- Pinhole camera, epipolar geometry, Fundamental/Essential matrix, Canny, Harris, SIFT, Optical Flow
- `01b-classical-descriptors.tex` --- Keypoint detectors (FAST/ORB), Lowe's ratio test, RANSAC iteration math, RAFT optical flow
- `02-cnn-foundations.tex` --- Convolution arithmetic, backprop derivation, BatchNorm vs LayerNorm, ResNet skip connection
- `02b-modern-convnets.tex` --- Depthwise separable convolutions, MobileNetV2 inverted bottlenecks, ShuffleNet MAC, ConvNeXt-V2 GRN
- `03-vision-transformers.tex` --- ViT architecture, configurations (Base to ViT-22B), DeiT distillation, Attention Distance, NaViT patch packing
- `04-hierarchical-vit.tex` --- Swin Transformer, W-MSA/SW-MSA cyclic shift masking, PVT Spatial Reduction, CoAtNet, FastViT
- `05-object-detection.tex` --- Two-stage detection (Faster R-CNN, FPN), RoIPool vs RoIAlign, CIoU loss, Focal Loss derivation
- `05b-single-stage-yolo.tex` --- SSD, RetinaNet, YOLOv1--v11 architectural lineage, Task-Aligned Assigner (TAL), DFL, YOLOv10 NMS-free
- `06-detr-query-detection.tex` --- DETR Hungarian bipartite set loss, Deformable Attention, DINO-DETR contrastive denoising, RT-DETR
- `07-segmentation-sam.tex` --- Semantic (U-Net, DeepLabv3+ ASPP), Instance (Mask R-CNN), Panoptic (Mask2Former), SAM foundation segmentation
- `07b-sam-foundation-seg.tex` --- SAM ViT-H windowed backbone, prompt encoder, ambiguity handling, SAM 2 streaming video memory bank
- `08-self-supervised-vision.tex` --- Contrastive learning (SimCLR InfoNCE MI lower bound, MoCo), non-contrastive (BYOL collapse proof)
- `08b-ssl-mae-dino.tex` --- MAE asymmetric 75% patch masking, DINO centering and sharpening, DINOv2 KoLeo regularizer, Register tokens
- `09-vlm-clip-siglip.tex` --- CLIP dual-encoder loss and gradients, distributed All-Gather memory issue, OpenCLIP, SigLIP sigmoid pairwise loss
- `10-multimodal-llms.tex` --- LLaVA architecture, AnyRes dynamic grid slicing, InternVL-2, Qwen2-VL native 2D/3D RoPE, POPE hallucination
- `10b-vlm-benchmarks-hallucination.tex` --- POPE random/popular/adversarial splits, Visual Contrastive Decoding (VCD), MMMU, DocVQA OCR
- `11-diffusion-foundations.tex` --- DDPM forward/reverse process, Score-based SDE/ODE, Tweedie's formula, EDM continuous $\sigma$, CFG guidance
- `12-latent-diffusion-dit.tex` --- Latent Diffusion Models (LDM), VAE $f=8$ perceptual compression, SDXL conditioning, DiT adaLN-Zero scaling
- `13-flow-matching-flux.tex` --- Flow Matching straight trajectories, velocity field objective, OT-CFM, Rectified Flow reflowing, Flux.1 12B
- `14-video-models.tex` --- 3D Conv, TimeSformer factorized space-time attention, 3D Causal VAE, Sora spacetime patch grid, VBench
- `15-3d-nerf-3dgs.tex` --- NeRF volumetric quadrature rendering, Instant-NGP spatial hashing, 3DGS covariance projection, tile rasterization
- `15b-3d-splatting-details.tex` --- 3D covariance parameterization, EWA Jacobian projection, Spherical Harmonics (degrees 0--3), 64-bit Radix sort
- `16-benchmarks-eval.tex` --- ImageNet-1K, COCO mAP, Panoptic Quality (SQ/RQ), POPE, FID 2-Wasserstein derivation, LPIPS perceptual loss
- `17-production-edge.tex` --- Conv-BN fusion derivation, INT8 PTQ KL divergence calibration, TensorRT plugins, ByteTrack Kalman tracking
- `18-system-design.tex` --- BEVFormer autonomous driving perception pipeline, ColPali late-interaction visual document retrieval RAG
- `19-code-blind.tex` --- Runnable PyTorch stubs: Conv2D im2col, ResNet Bottleneck, ViT PatchEmbed, 2D RoPE, InfoNCE, Vectorized NMS
- `19b-training-loops.tex` --- Execution loops: SimCLR step, DDPM step, Rectified Flow Matching step, AnyRes dynamic image slicer
- `20-napkin-math.tex` --- Conv FLOPs, ViT quadratic attention scaling, receptive field recurrence, VLM AnyRes tokens, 3DGS VRAM
- `20b-napkin-scaling.tex` --- Factorized video attention complexity, ViT patch size shrinkage laws, learning rate scaling rules, NPU TOPS
- `21-debugging.tex` --- NaN box regression losses, ViT small data failure, Diffusion gray mush, 3DGS floater needles, VLM visual token loops
- `22-rapid-glossary.tex` --- 20 rapid-fire questions, 20 essential glossary definitions
- `22b-rapid-fire-drill.tex` --- 30 rapid-fire technical questions covering entire vision stack, flashcards, and edge case traps
- `23-formulas.tex` --- Master vision formula sheet covering all 24 architectural mechanisms
- `24-seminal-papers.tex` --- 30 seminal vision papers card (AlexNet to Flux.1 and Sora)
- `25-canon-references.tex` --- Canonical vision bibliography arranged by section with arXiv IDs
- `26-study-plan.tex` --- 21-day revision protocol, phase breakdown, and interview red flags
