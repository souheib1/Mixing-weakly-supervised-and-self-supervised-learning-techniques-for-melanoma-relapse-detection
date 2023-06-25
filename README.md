# Mixing weakly-supervised and self supervised-learning techniques for melanoma relapse detection

This project focuses on the VisioMel Challenge whose goal is predicting melanoma relapse. It is a complex and often unreliable task, with diagnostic accuracy often varying among experienced medical professionals. Recent advancements in SSL and WSL offer promising new solutions for improving the accuracy of cancer relapse detection. 

![Capture](https://github.com/souheib1/Mixing-weakly-supervised-and-self-supervised-learning-techniques-for-melanoma-relapse-detection/assets/73786465/96758827-183d-4669-8c2b-9f3c27deceb6)

In this work, we specifically explore combinations of SSL and WSL techniques, namely **BYOL** (Bootstrap Your Own Latent) followed by **AbMIL** (Attention-based Multiple Instance Learning) and **SimCLR** (Simple Framework for Contrastive Learning of Representations) followed by **AbMIL**. BYOL and SimCLR are SSL techniques that train models using unlabeled data, while AbMIL is a WSL approach that exploits weakly labeled data without pixel-level annotations. 

By leveraging the strengths of BYOL and SimCLR for self-supervised learning, as well as AbMIL for weakly supervised learning, our research provides insight into the effectiveness of using these techniques in the context of melanoma relapse prediction.

![PIPELINE](https://github.com/souheib1/Mixing-weakly-supervised-and-self-supervised-learning-techniques-for-melanoma-relapse-detection/assets/73786465/2b83eee7-cf54-475a-b354-cc3e829567ac)


### References
[1] Ting Chen et al. “A Simple Framework for Contrastive Learning of Visual Representations”. In:
ArXiv abs/2002.05709 (2020).

[2] Jean-Bastien Grill et al. “Bootstrap your own latent-a new approach to self-supervised learning”.
In: Advances in neural information processing systems 33 (2020), pp. 21271–21284. [Link to the paper](https://arxiv.org/pdf/2006.07733.pdf)

[3] Maximilian Ilse, Jakub M. Tomczak, and Max Welling. Attention-based deep multiple instance
learning. June 2018. [Link to the paper]( https://arxiv.org/abs/1802.04712)

[4] Bin Li, Yin Li, and Kevin W. Eliceiri. Dual-stream multiple instance learning network for whole
slide image classification with self-supervised contrastive learning. Apr. 2021. [Link to the paper](https://arxiv.org/abs/2011.08939)

[5] Ming Y. Lu et al. Data efficient and weakly supervised computational pathology on whole slide
images. May 2020. [Link to the paper](https://arxiv.org/abs/2004.09666)
