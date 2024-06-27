FaceRecognition project with Deep Convolutional Neural Networks using PyTorch and Scikit-learn libraries.
Dataset : Casia-WebFace, reduced to top 500 classes by number of instances. 
Number of instances is 129902.

Comparing the results from different architectures:
- CNN;
- ResNet18;
- ResNet50.

Also ML models:
- ZeroRule;
- NaiveBayes;
- SupportVectorMachine.

For ML models I have used PCA to reduce images to 150 components.
ResNet models are unfrozen from layer3 and on.
