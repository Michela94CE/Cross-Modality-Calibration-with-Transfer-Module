# Cross-Modality-Calibration-with-Transfer-Module

This is the repository for the paper "Cross-Modality Calibration in Multi-Input Network for Axillary Lymph Node Metastasis Evaluation"

# Abstract
The use of deep neural networks (DNNs) in medical images has enabled the development of solutions characterized by the need of leveraging information coming from multiple sources, raising the Multimodal Deep Learning. DNNs are known for their ability to provide hierarchical and high-level representations of input data. This capability has led to the introduction of methods performing data fusion at an intermediate level, preserving the distinctiveness of the heterogeneous sources in modality-specific paths, while learning the way to define an effective combination in a shared representation. However, modeling the intricate relationships between different data remains an open issue. In this paper, we aim to improve the integration of data coming from multiple sources. We introduce between layers belonging to different modality-specific paths a Transfer Module (TM) able to perform the cross-modality calibration of the extracted features, reducing the effects of the less discriminative ones. As case of study, we focus on the axillary lymph nodes metastasis evaluation in malignant breast cancer, a crucial prognostic factor, affecting patient's survival. We propose a Multi-Input Single-Output 3D Convolutional Neural Network (CNN) that considers both images acquired with multiparametric Magnetic Resonance and clinical information. In particular, we assess the proposed methodology using four architectures, namely BasicNet and three ResNet variants, showing the improvement of the performance obtained by including the TM in the network configuration. Our results achieve up to 90\% and 87\% of accuracy and Area under ROC curve, respectively when the ResNet10 is considered, surpassing various fusion strategies proposed in the literature.

# Impact Statement
In the context of breast cancer, the metastatic involvement of axillary lymph nodes (ALN) stands out as a crucial prognostic factor, reflecting the intrinsic behavior of the primary tumor. Multiparametric Magnetic Resonance Imaging (MRI) enables a comprehensive examination, providing both physiological and morphological characteristics through sequences involving pre-contrast and post-contrast agent administration. This highlights the necessity of integrating diverse information, particularly when considering histological data in conjunction with images. However, current state-of-art solutions typically exploit features extracted from the post-contrast series, neglecting the use of the others. The methodology presented in this paper harnesses Multimodal Deep Learning (MDL) to overcome this limitation, efficiently integrating clinical information and features from multiple image modalities. Demonstrating an accuracy of 90% in the metastatic evaluation of ALN, our algorithm has the potential to support radiologists in their daily analysis of breast MRI.

# Transfer Module

- We propose an innovative Transfer Module (TM) that can be inserted between layers belonging to different modality-specific paths to model the complex interaction of heterogeneous data.
- We make our TM able to modify the extracted features maps in each modality-specific path taking into account the complementary nature of the inputs. We will refer to this procedure as cross-modality calibration.
- We implement in the TM a specific operation, denoted as gating mechanism, that reduces the importance of the least discriminative characteristics in each modality-specific features maps exploiting the descriptors of all the inputs.  
- We include the TM in the training process, making the definition of the most relevant features well-suited for the specific task to solve.

![Alt text](https://github.com/Michela94CE/Cross-Modality-Calibration-with-Transfer-Module/tree/main/img/tm.png)
  

# Authors
Michela Gravina, Domiziana Santucci, Ermanno Cordelli, Paolo Soda, and Carlo Sansone

- Michela Gravina and Carlo Sansone are with the Department of Electrical Engineering and Information Technology (DIETI) of the University of Naples Federico II, Via Claudio 21, 80125, Naples, Italy (email:michela.gravina@unina.it, carlo.sansone@unina.it).

- Domiziana Santucci is with the Department of Radiology, University of Rome “Campus Bio-Medico”, Via Alvaro del Portillo, 21, 00128 Rome, Italy (e-mail: d.santucci@policlinicocampus.it).

- Ermanno Cordelli is with the Unit of Computer Systems and Bioinformatics, Dept. of Engineering, University of Rome Campus Bio-Medico, via Alvaro del Portillo 21, 00128, Roma, Italy (email: e.cordelli@unicampus.it).

- Paolo Soda is with the Department of Diagnostics and Intervention, Radiation Physics, Biomedical Engineering, Umea University, Universitetstorget 4, 90187, Umea, Sweden, and with the Unit of Computer Systems and Bioinformatics, Dept. of Engineering, University of Rome Campus Bio-Medico, via Alvaro del Portillo 21, 00128, Roma, Italy (email: p.soda@unicampus.it)
