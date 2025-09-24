# audio-ml
Collection of machine learning notebooks and experiments demonstrating skills in PyTorch, TensorFlow, and deep learning for audio.

1. (ESC-50) From scratch CNN on mel spectrograms

2. (ESC-50) Pretrained VGGish embeddings + lightweight classifier

| Aspect               | Script 1 (CNN)                                             | Script 2 (VGGish)                                                                                                        |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Framework            | PyTorch                                                    | TensorFlow + TensorFlow Hub                                                                                              |
| Model Type           | Custom CNN trained from scratch                            | Pretrained **VGGish** embeddings + small dense classifier                                                                |
| Input Representation | Log-mel spectrograms computed manually for each audio clip | Log-mel spectrograms are implicitly computed by VGGish; the script uses VGGish embeddings (fixed-length vector per clip) |
| Feature Extraction   | Hand-crafted in dataset class using `librosa`              | Uses pretrained VGGish model to extract features, no convolution training on raw audio                                   |
| Training             | Full end-to-end CNN                                        | Only the small dense classifier on top of frozen embeddings                                                              |

