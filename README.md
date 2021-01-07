# FeatureNet
This is a re-implementation of FeatureNet in Tensorflow 2. FeatureNet is a deep learning architecture for machining feature recognition that utilises a voxel representation and 3D CNN.

The code is based on this original [paper](https://www.sciencedirect.com/science/article/abs/pii/S0010448518301349). This paper's original code can be found [here](https://github.com/zibozzb/FeatureNet).

![featurenet_network](imgs/featurenet.png)

## Requirements
- Python
- Tensorflow
- Numpy
- h5py
- Binvox <sup>(Opensource software)</sup>


## Citation
Original paper citation:

    @article{featurenet2018,
      Author = {Zhibo Zhang, PrakharJaiswal, Rahul Rai},
      Journal = {Computer-Aided Design},
      Title = {FeatureNet: Machining feature recognition based on 3D Convolution Neural Network},
      Year = {2018}
    }

    @article{tensorflow2featurenet,
      Author = {Andrew R Colligan},
      Title = {Tensorflow 2 FeatureNet},
      Journal = {https://gitlab.com/qub_femg/machine-learning/featurenet-tensorflow-2},
      Year = {2021}
}
