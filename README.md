# SENSE-Edge  
Sensor-to-Image Neuro-Symbolic & Vision-Language Framework for Explainable Environmental Intelligence on Edge Devices

---

## Overview

SENSE-Edge is an agentic, neuro-symbolic, vision–language framework designed for real-time, explainable environmental monitoring on edge devices.

The system converts heterogeneous environmental sensor data (air, water, climate, and indoor) into structured visual representations. These images are processed by Vision-Language Models (VLMs), mapped to symbolic knowledge, and fed into a probabilistic reasoning layer. The final output is an agentic system capable of taking safe, explainable real-time actions.

---

## 1. Unified Architecture

The following figure illustrates the full pipeline from sensors to edge actions.

![SENSE-Edge Architecture](docs/images/architecture_overview.png)

This architecture integrates:

- Sensor-to-image encoding (GAF, RP, Spectrogram, Heatmaps)  
- Neural VLM perception  
- Symbolic ontology-based reasoning  
- Agentic decision-making  
- Real-time explainable actions on edge devices  

---

## 2. Neuro-Symbolic Reasoning Pipeline

This pipeline explains how SENSE-Edge combines neural embeddings with symbolic knowledge to produce explainable, policy-aware outputs.

![Neuro-Symbolic Pipeline](docs/images/neurosymbolic_pipeline.png)

Components involved:

- VLM produces embeddings  
- Symbolic embedding maps neural features to ontology terms  
- Probabilistic reasoning combines rules + uncertainty  
- Decision module outputs confidence scores + explanations  
- Regulatory transparency constraints ensure safety on edge devices  

---

## 3. Agentic AI Layer

The agentic layer orchestrates autonomous decision-making across monitoring, task execution, and explainability.

![Agentic AI](docs/images/agentic_ai.png)

This layer includes:

- Task agents  
- Monitoring agents  
- Explainability agents  
- Safety and regulatory constraints  
- Closed-loop feedback based on environmental changes  

---

## Objectives

- Transform sensor streams into 2D image representations  
- Apply VLMs for multimodal environmental perception  
- Add ontology-based symbolic reasoning for structure and logic  
- Enable autonomous decision-making on edge devices  
- Provide full transparency and explainability  

---

## Proof of Concept

**Dataset:** Beijing PM2.5 (5-class classification)

- Time-series windows were converted into images  
- Adapted CLIP-style VLM was used  
- Achieved: **41% accuracy**  
- Comparable with published results (~48.8%)  

This validates the Sensor-to-Image → VLM → Symbolic Reasoning approach.

---

## Repository Structure

```
SENSE-Edge/
│
├── data/
│   └── (datasets and generated image encodings)
│
├── encoding/
│   └── Sensor-to-image transformation scripts (GAF, RP, Spectrogram, etc.)
│
├── vlm/
│   └── CLIP / ViT training and inference modules
│
├── reasoning/
│   └── Neuro-symbolic reasoning, ontology, ASP logic
│
├── agent/
│   └── Agentic decision-making and safety modules
│
├── experiments/
│   └── Evaluation scripts and reproducible experiments
│
├── docs/
│   └── images/ (architecture diagrams, figures)
│
└── requirements.txt
```

---

## Status

| Component                      | Status      |
|-------------------------------|-------------|
| Architecture defined          | ✔ Done      |
| Image encoding implemented    | ✔ Done      |
| CLIP prototype tested         | ✔ Done      |
| Neuro-symbolic reasoning      | In Progress |
| Agentic AI                    | In Progress |
| Edge deployment               | In Progress |

---

## Author

**Yousef Alhattab**  
PhD Researcher – AI Engineering  
Environmental Intelligence • Edge AI • Neuro-Symbolic Systems

---

If you want, I can also prepare:

- A polished academic abstract  
- A full GitHub Pages website  
- A PDF project overview for supervisors  
- A thesis chapter based on this architecture  

Just tell me.
