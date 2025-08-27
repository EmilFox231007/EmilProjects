# Computer Vision: Root Length Detection and Segmentation 

### This first part of the project uses Computer Vision and Machine Learning techniques to segment and analyze plant root structures from petri dish images and masks and ultimately obtaining the root length measurements and root tip coordinates for faster and automatic inoculation of the plants, in which I also performed a robotics simulation to inoculate the root tips in a simulated environment. ML U-Net based models were also trained with plant images and masks. 

> TL;DR: Feed an image → get a CSV/JSON of `(x, y)` root-tip coordinates. Trainable end-to-end, deployable to Azure ML, and scriptable via CLI or Python.

## Project Overview:

- **Input:** Images of plants grown on plates.

- **Process:** Preprocessing → segmentation → morphology & graph analysis → coordinate extraction.

- **Output:** Root tip coordinates for each plant root in the image.

The pipeline was developed as part of a broader effort to automate plant phenotyping tasks, providing a reproducible and modular approach for image-based root analysis.

## Features

- **End-to-end pipeline**: preprocessing → segmentation → post-processing → root-tip coordinates  
- **Patch-based U-Net** (Keras/TensorFlow) for high-res images  
- **Training utilities**: automatic cropping/padding & patch extraction  
- **Evaluation**: accuracy/F1 + per-image reports  