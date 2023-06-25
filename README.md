# Mixing weakly-supervised and self supervised-learning techniques for melanoma relapse detection

This project focuses on the VisioMel Challenge whose goal is predicting melanoma relapse. It is a complex and often unreliable task, with diagnostic accuracy often varying among experienced medical professionals. Recent advancements in SSL and WSL offer promising new solutions for improving the accuracy of cancer relapse detection. 

In this work, we specifically explore combinations of SSL and WSL techniques, namely **BYOL** (Bootstrap Your Own Latent) followed by **AbMIL** (Attention-based Multiple Instance Learning) and **SimCLR** (Simple Framework for Contrastive Learning of Representations) followed by **AbMIL**. BYOL and SimCLR are SSL techniques that train models using unlabeled data, while AbMIL is a WSL approach that exploits weakly labeled data without pixel-level annotations. 

By leveraging the strengths of BYOL and SimCLR for self-supervised learning, as well as AbMIL for weakly supervised learning, our research provides insight into the effectiveness of using these techniques in the context of melanoma relapse prediction.

![PIPELINE](https://github.com/souheib1/Mixing-weakly-supervised-and-self-supervised-learning-techniques-for-melanoma-relapse-detection/assets/73786465/2b83eee7-cf54-475a-b354-cc3e829567ac)
