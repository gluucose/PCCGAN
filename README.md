# Image2Points: A 3D Point-based Context Clusters GAN for High-Quality PET Image Reconstruction (ICASSP 2024)
Paper link: [https://ieeexplore.ieee.org/document/10446360](https://ieeexplore.ieee.org/document/10446360)

[New] The extension of this work has been accepted by IEEE TCSVT.
Paper link: [https://ieeexplore.ieee.org/abstract/document/10526270](https://ieeexplore.ieee.org/abstract/document/10526270)

## Abstract 
To obtain high-quality Positron emission tomography (PET) images while minimizing radiation exposure, numerous methods have been proposed to reconstruct standard-dose PET (SPET) images from the corresponding low-dose PET (LPET) images. However, these methods heavily rely on voxel-based representations, which fall short of adequately accounting for the precise structure and fine-grained context, leading to compromised reconstruction. In this paper, we propose a 3D point-based context clusters GAN, namely PCC-GAN, to reconstruct high-quality SPET images from LPET. Specifically, inspired by the geometric representation power of points, we resort to a point-based representation to enhance the explicit expression of the image structure, thus facilitating the reconstruction with finer details. Moreover, a context clustering strategy is applied to explore the contextual relationships among points, which mitigates the ambiguities of small structures in the reconstructed images. Experiments on both clinical and phantom datasets demonstrate that our PCC-GAN outperforms the state-of-the-art reconstruction methods qualitatively and quantitatively. 

![image](https://github.com/gluucose/PCCGAN/assets/55613873/0726d007-e6b9-4234-8b6f-1ec45dd076eb)
<p align="center">Fig. 1. Overview of the proposed PCC-GAN.</p>

