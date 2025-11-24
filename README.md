# SENSE-Edge

**Sensor-to-Image Neuro-Symbolic & Vision-Language Framework for Explainable Environmental Intelligence on Edge Devices**

---

## Overview

SENSE-Edge is an agentic, neuro-symbolic, vision–language framework for real-time, explainable environmental monitoring on edge devices.

It converts heterogeneous sensor data (air, water, climate, indoor) into visual representations compatible with large vision models.

Key components:
- Sensor-to-Image Encoding (GAF, RP, Spectrograms, Heatmaps)
- Vision-Language Models (CLIP / ViT)
- Neuro-Symbolic Reasoning (Ontology + ASP)
- Agentic Autonomous Decision-Making
- Edge-Optimised Deployment (Jetson)

---

## Objectives

- Transform sensor streams into 2D images
- Apply VLMs for perception
- Add ontology-based symbolic reasoning
- Enable autonomous environmental monitoring
- Ensure explainability on edge devices

---

## Architecture

Sensors → Encoding → VLM → Symbolic Mapping → ASP Reasoning → Agent → Explainable Output

---

## Proof of Concept

Dataset: **Beijing PM2.5**

- Converted sensor windows to images  
- Used adapted CLIP model  
- Accuracy: **41% (5-class)**  
- Comparable to reported ~48.8%

---

## Repository Structure

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

yaml
Copy code

---

## Status

- Architecture defined ✅
- Encoding implemented ✅
- CLIP prototype tested ✅
- Reasoning extended 🔄
- Edge deployment 🔄

---

## Author

**Yousef Alhattab**  
PhD Researcher – AI Engineering  
Focus: Environmental Intelligence | Edge AI | Neuro-Symbolic Systems
