

# Visualization:
![image](https://github.com/Udrs/DDPM-based-Change-Detection/blob/main/inference_vis_video/output_video2.gif)
![image](https://github.com/Udrs/DDPM-based-Change-Detection/blob/main/inference_vis_video/output_video4.gif)
---
![image](https://github.com/Udrs/DDPM-based-Change-Detection/blob/main/inference_vis_video/output_video5.gif)
![image](https://github.com/Udrs/DDPM-based-Change-Detection/blob/main/inference_vis_video/output_video6.gif)

See our other work within：https://github.com/sstary/SSRS

# Motivations:
Most existing methods are ineffective in simultaneously capturing long-range dependencies and exploiting local spatial information, making it challenging to obtain fine-grained and accurate CD maps. To overcome these obstacles, a novel Denoising Diffusion Probabilistic Model (DDPM)-based generative CD approach called GCD-DDPM is proposed for remote sensing data.

# Contributions:
1. The proposed GCD-DDPM is a pure generative model tailored for CD tasks. By utilizing an adaptive calibration approach, the GCD-DDPM excels in gradually learning and refining data representations, effectively distinguishing diverse changes in natural scenes and urban landscapes.
2. The GCD-DDPM introduces differences among multi-level features extracted from pre- and post-change images within DCE, which are then integrated into the sampling process to guide the generation of CD maps. This method allows for a more fine-grained capture of changes.
3. An NSSE is proposed by employing an attention mechanism to suppress noise in the difference information derived in the current step. This process is vital for aiding the DCE in extracting more accurate change-aware representations and enhances the DCE's ability to distinguish and capture changes.

# Overall Architecture:
![image](https://github.com/udrs/GCD/assets/71435435/a4f04b4c-9700-4bbf-b147-7845345b4532)

# comparison 
(a) DDPM-CD and (b) the proposed GCD-DDPM.

![image](https://github.com/udrs/GCD/assets/71435435/30bdf8d5-3675-4c21-b057-1a6caebeddd5)
