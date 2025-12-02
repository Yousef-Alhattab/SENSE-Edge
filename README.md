# SENSE-Edge

**Sensor-to-Image Neuro-Symbolic & Vision-Language Framework for Explainable Environmental Intelligence on Edge Devices**

---

## Overview

SENSE-Edge is an agentic, neuro-symbolic, vision–language framework for real-time, explainable environmental monitoring on edge devices.

It converts heterogeneous sensor data (air, water, climate, indoor) into visual representations compatible with large vision models.

### Key Components

- Sensor-to-Image Encoding (GAF, RP, Spectrograms, Heatmaps)  
- Vision-Language Models (CLIP / ViT)  
- Neuro-Symbolic Reasoning (Ontology + ASP)  
- Agentic Autonomous Decision-Making  
- Edge-Optimised Deployment (NVIDIA Jetson)

---

## Objectives

- Transform sensor streams into 2D images  
- Apply VLMs for perception  
- Add ontology-based symbolic reasoning  
- Enable autonomous environmental monitoring  
- Ensure explainability on edge devices  

---

## Architecture

```
Sensors 
    → Sensor-to-Image Encoding 
        → Vision-Language Model (CLIP / ViT) 
            → Symbolic Mapping 
                → ASP Reasoning 
                    → Agent 
                        → Explainable Output
```

---

## Proof of Concept

**Dataset:** Beijing PM2.5 (5-class classification)

- Converted environmental sensor windows into image representations  
- Used an adapted CLIP-style VLM  
- Achieved 41% accuracy (5 classes)  
- Comparable to reported ~48.8% in literature  

---

## Repository Structure

```
SENSE-Edge/
│
├── data/
├── encoding/
├── vlm/
├── reasoning/
├── agent/
├── experiments/
├── docs/
└── requirements.txt
```

---

## Status

| Component | Status |
|----------|--------|
| Architecture defined | ✅ |
| Image encoding implemented | ✅ |
| CLIP prototype tested | ✅ |
| Neuro-symbolic reasoning | 🔄 In progress |
| Edge deployment | 🔄 In progress |

