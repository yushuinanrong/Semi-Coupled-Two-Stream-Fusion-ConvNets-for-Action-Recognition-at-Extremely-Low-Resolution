# Semi-Coupled-Two-Stream-Fusion-ConvNets-for-Action-Recognition-at-Extremely-Low-Resolution

We leveraged deep learning techniques and proposed multiple, end-to-end ConvNets for action recognition from extremely Low Resolution (eLR) videos (e.g., 16 × 12 pixels). We proposed multiple eLR ConvNet architectures, each leveraging and fusing spatial and temporal information. Further, in order to leverage high resolution (HR) videos in training, we incorporated eLR-HR coupling to learn an intelligent mapping between the eLR and HR feature spaces. The effectiveness of this architecture has been validated on two public datasets on which our algorithms have outperformed state-of-the-art methods.

# Datasets:

we conducted experiments on two publiclyavailable video datasets.
We first used the ROI sequences from the multi-view IXMAS action dataset, where each
subject occupies most of the field of view. This dataset
includes 5 camera views, 12 daily-life motions each performed
3 times by 10 actors in an indoor scenario. To generate the eLR videos (thus eLR-IXMAS), we decimated the original frames to
16×12 pixels and then upscaled them back to 32×32 pixels
by bi-cubic interpolation. On the other hand, we generate the HR data by decimating the original frames straight to
32 × 32 pixels. We perform leave-person-out cross validation
in each case and compute correct classification rate (CCR) and standard deviation (StDev) to measure performance.

The second dataset is HMDB51 dataset. The HMDB dataset
consists of 6,849 videos divided into 51 action categories. We followed the same way to generate the eLR and HR versions as described above.

You can download the original IXMAS ROI sequences use the link: http://cvlab.epfl.ch/data/ixmas10

For HMDB, you can download the dataset use the link: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

# ConvNet Library

We implemented the models and conducted experiments in the Matconvnet toolbox (1.0-beta23).
You can download the toolbox use the link: http://www.vlfeat.org/matconvnet/.

Please follow the toolbox instruction to install and compile. 

# Optical flow

We computed the colored optical flow using Dr.Chi Liu's toolbox. You can download it use the link: https://people.csail.mit.edu/celiu/OpticalFlow/

Please note we use the default parameter setting provided in the examplary script of the toolbox.

# Reference

For more details, please refer to our paper:

J. Chen, J. Wu, J. Konrad, and P. Ishwar, “Semi-Coupled Two-Stream Fusion ConvNets for Action Recognition at Extremely Low Resolutions,” in Proc. 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), Mar. 2017.
