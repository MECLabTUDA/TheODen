# Semi-supervised Learning
This section shows how to run a semi-supervised pre-training task. 
This demo implements the SimCLR pre-training proposed by the paper [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html) published at ICML 2020. 
The framework pre-trains an encoder on a large pool of unlabeled data, and fine-tunes the segmentation model on a smaller pool of annotated data. 
## Pre-training
The pre-training step can be performed by starting the server tailored for pre-training using `python ssl_server.py`. After starting the server, the clients can be started using `python client.py`.
## Fine-tuning
Once pre-training has finished, the second server managing the fine-tuning can be started using `python fine_server.py`. The clients are started as above using `python client.py`.
