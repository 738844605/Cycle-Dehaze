#Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing
## Prerequisites

* TensorFlow-gpu 1.9.0
* Tensorboard 1.9.0
* Python 3.5.6
* Opencv-python 4.0.0.21
## FILE STRUCTURE
```
Cycle-Dehaze-master
|-- README.md
|-- X
    |-- A|-- *.png
|-- Y
    |-- B|-- *.png
|-- XOUT
    |-- x.tfrecords
|-- YOUT
    |-- y.tfrecords
|-- pretrained
    |-- *.pb
|-- checkpoints
|-- samples
    |-- *.png
```
## Data preparing

* First, download a hazy dataset, e.g. NTIRE 2018/2020 or RESIDE
* Hazy images in X/A for training, Clear images in Y/B for training
* Write the dataset to tfrecords
```
python build_data.py --X_input_dir X/A --X_output_file XOUT/x.tfrecords --Y_input_dir Y/B --Y_output_file YOUT/y.tfrecords
```

Check `python build_data.py --help` for more details.

## Training

```
python train.py --X=XOUT/x.tfrecords --Y=YOUT/y.tfrecords
```

If you halted the training process and want to continue training, then you can try:
```
python train.py --load_model 20191123-1530
```

## Check TensorBoard to see training progress and generated images

```
tensorboard --logdir checkpoints/20191123-1530
```
## Export model
You can export from a checkpoint to a standalone GraphDef file as follow:

```
python export_graph.py --checkpoint_dir checkpoints/${datetime} \
                          --XtoY_model apple2orange.pb \
                          --YtoX_model orange2apple.pb \
                          --image_size 256
```
```
e.g. python export_graph.py --checkpoint_dir checkpoints/20191205-1022 --XtoY_model x2y.pb --YtoX_model y2x.pb --image_size [256,256]
```

## Inference and upsamling
After exporting model, you can use it for inference. For example:

```
python inference.py --model pretrained/Hazy2GT_indoor.pb \
                     --input input_haze.jpg \
                     --output output_dehaze.jpg \
                     --image_size 256
```

Use matlab code to improve the resolution of results:
```
laplacian.m
```
## You also can test your own images:
```
python test.py
```
## Here is the list of arguments:
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                [--use_lsgan [USE_LSGAN]] [--nouse_lsgan]
                [--norm NORM] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2]
                [--learning_rate LEARNING_RATE] [--beta1 BETA1]
                [--pool_size POOL_SIZE] [--ngf NGF] [--X X] [--Y Y]
                [--load_model LOAD_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size, default: 1
  --image_size IMAGE_SIZE
                        image size, default: 256
  --use_lsgan [USE_LSGAN]
                        use lsgan (mean squared error) or cross entropy loss,
                        default: True
  --nouse_lsgan
  --norm NORM           [instance, batch] use instance norm or batch norm,
                        default: instance
  --lambda1 LAMBDA1     weight for forward cycle loss (X->Y->X), default: 10.0
  --lambda2 LAMBDA2     weight for backward cycle loss (Y->X->Y), default:
                        10.0
  --learning_rate LEARNING_RATE
                        initial learning rate for Adam, default: 0.0002
  --beta1 BETA1         momentum term of Adam, default: 0.5
  --pool_size POOL_SIZE
                        size of image buffer that stores previously generated
                        images, default: 50
  --ngf NGF             number of gen filters in first conv layer, default: 64
  --X X                 X tfrecords file for training, default:
                        data/tfrecords/apple.tfrecords
  --Y Y                 Y tfrecords file for training, default:
                        data/tfrecords/orange.tfrecords
  --load_model LOAD_MODEL
                        folder of saved model that you wish to continue
                        training (e.g. 20170602-1936), default: None
build_data.py:
  --X_input_dir: X input directory, default: data/apple2orange/trainA
    (default: 'data/apple2orange/trainA')
  --X_output_file: X output tfrecords file, default:
    data/tfrecords/apple.tfrecords
    (default: 'data/tfrecords/apple.tfrecords')
  --Y_input_dir: Y input directory, default: data/apple2orange/trainB
    (default: 'data/apple2orange/trainB')
  --Y_output_file: Y output tfrecords file, default:
    data/tfrecords/orange.tfrecords
    (default: 'data/tfrecords/orange.tfrecords')
```

### Notes
* If high constrast background colors between input and generated images are observed (e.g. black becomes white), you should restart your training!
* Train several times to get the best models.

## Pretrained models
My pretrained models are available at https://github.com/vanhuyz/CycleGAN-TensorFlow/releases

## Contributing
Please open an issue if you have any trouble or found anything incorrect in my code :)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

* CycleGAN paper: https://arxiv.org/abs/1703.10593
* Official source code in Torch: https://github.com/junyanz/CycleGAN

## License
This project is licensed under the MIT License - see the <a href="https://github.com/engindeniz/Cycle-Dehaze/blob/master/LICENSE">LICENSE</a> file for details.
