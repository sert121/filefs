# Model Card: Diffusion-v1

## Model Details
- Architecture: UNet + Time Embeddings
- Dataset: CelebA-HQ + Custom Portrait Dataset (120k images)
- Training Steps: 900k on 8x A100 GPUs
- FID: 3.2

## Intended Use
- Conditional image generation (sketch-to-face)
- Not for use in medical imaging without fine-tuning

## Ethical Considerations
- Potential misuse for deepfakes
- Dataset bias towards Western facial features

## Limitations
- Struggles with non-frontal images
- Fails to preserve fine-grained hair textures in high noise conditions
