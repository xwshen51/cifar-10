# Image Classification on CIFAR-10


## Environment
- Pythoy 3.7.3
- PyTorch 1.2.0

## Accuracy
| Model             | Test Acc.   |
| ----------------- | ----------- |
| EfficientNet-B4   | 98.23%      |
| EfficientNet-B0   | 97.58%      |
| DenseNet-Distill  | 95.91%      |
| DenseNet (BC-Wide)| 95.43%      |

## Reproduce the results
* DenseNet (1 GPU with 11GB):
`python main.py --model 'DenseNetWide'`
* DenseNet-Distill (2 GPUs): `python distill.py --model 'DenseNetWide'`
* EfficientNet-B0 (2 GPUs): `python main.py --model 'EfficientNetB0' --lr 0.01`
* EfficientNet-B4 (4 GPUs): `python main.py --model 'EfficientNetB4' --lr 0.01 --decay .5 --decay_epoch 30 --batch_size 64`

Note: When training EfficientNet-B4, I forgot to save the random seed, so the reproducing result may not be exactly the same (could be better). Others should be.

## Load the trained model for testing

The trained model is saved in folder 'saved_model' with logging files. Please refer to the codes for reloading models from checkpoints in `main.py`, and use the `test` function to get the test accuracy.
