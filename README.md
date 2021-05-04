# GeekLearning
A repository of a project for the course Machine Learning at Aalto University. 
The goal is to *classify* images into three classes: **Anime, Cartoon, and Pokemon**. 
See the documentation in `docs/report.pdf` for more information. 
Only the report was submitted to the course. Models not described in the report 
are recent additions.

[Check out how models perform here](https://geeklearning.herokuapp.com/)

Models were trained with 64x64 RGB images. In particular, the ResNet-50 was trained with the 
transfer-learning technique (using pretrained weights as a starting point). The details of ResNet-50 training 
is as follows. The original ImageNet-optimized fully-connected layer was replaced with a new head with 3 outputs 
representing the 3 classes. In phase 1, only this new head was trained. In phase 2, more than half of the weights were updated.
See `src/train_models.py` for more information on how models were trained.

For inference, the models were [traced](https://pytorch.org/docs/stable/jit.html) to reduce the file size of saved models. 

The Flask application is available as a Docker image, which can be found here. Note that the image is large in size (~2GB).

Ideas to improve the project (arranged in decreasing order of importance):

* Reduce the size of the Docker image by moving model inference into C++ (OpenCV and LibTorch)

* Use OpenCV instead of relying on PIL (to pre-process images)

* Introduce dimensionality reduction models: PCA, autoencoders, etc., to assist classical ML methods (SVM)

* Implement a kernel-based SVM and other classical ML methods

* Add more deep learning models: EfficientNets, Transformers, original designs, etc.

* Re-train models with larger images (e.g., 224x224 images).

* For the web-application, move away from Python and use other lightweight frameworks (preferably with C++ backend).

* Move away from Heroku and use providers with GPU-enabled runtimes. Scale up the application.
