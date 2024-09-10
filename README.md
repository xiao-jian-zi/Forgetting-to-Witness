# Forgetting to Witness: Efficient Federated Unlearning and Its Visible Evaluation
PyTorch code for the paper "Forgetting to Witness: Efficient Federated Unlearning and Its Visible Evaluation".
## 1 Requirements
We recommended the following dependencies.

* python==3.8
* pytorch==1.11.0
* torchvision==0.12.0
* numpy==1.22.4

For more recommended dependencies, please refer to the file [`requirements.txt`]([https://github.com/xiao-jian-zi/MU-Goldfish-An-Efficient-Federated-Unlearning-Framework/blob/main/requirements.txt](https://github.com/xiao-jian-zi/Forgetting-to-Witness/blob/main/requirements.txt)).

## 2 How to use
### 2.1 Training new models
We use the CIFAR-10 dataset as an example.

Run `train_cifar10.py` to obtain the trained models: 

```bash
python train_cifar10.py
```
 You can specify the number of training epochs by setting the `epoch_num` in the file. If necessary, please modify the path to the dataset; otherwise, the code will automatically download the dataset to the default path when executed.

### 2.2 Unlearning
Run `unlearning_cifar10.py` to perform the unlearning process: 
```bash
python unlearning_cifar10.py
```
To customize the unlearning process, please make the following adjustments in the file:
* Modify the `epochs` value to specify the number of unlearning epochs.
* Modify the `your_backdoored_model_path` to set the path for the backdoored model.
* Modify the `forget_Proportion` to define the size of the forgetting dataset.
  
These configurations will allow you to tailor the unlearning process to your specific requirements. Ensure that the values are correctly set before running the training script to avoid any errors or unexpected results.

### 2.3 Visible Evaluation
We use the MNIST dataset as an example.

Run `GAN-MNIST.py` to perform the visible evaluation process: 

```bash
python GAN-MNIST.py 
```
To customize the evaluation process, please make the following adjustments in the file:
* Modify the `num_epochs` value to specify the number of unlearning epochs.
* Modify the `classifier_path` to the trained classifier checkpoint.
* Modify the `generator_save_path` to the save path for generator checkpoint.

These configurations will allow you to tailor the evaluation process to your specific requirements. Ensure that the values are correctly set before running the training script to avoid any errors or unexpected results.

## 3 Code Reference
For detailed code explanations and best practices, please refer to
* [https://github.com/akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
* [https://github.com/IMoonKeyBoy/The-Right-to-be-Forgotten-in-Federated-Learning-An-Efficient-Realization-with-Rapid-Retraining](https://github.com/IMoonKeyBoy/The-Right-to-be-Forgotten-in-Federated-Learning-An-Efficient-Realization-with-Rapid-Retraining)
* [https://github.com/vikram2000b/bad-teaching-unlearning](https://github.com/vikram2000b/bad-teaching-unlearning)
* [https://github.com/bboylyg/NAD](https://github.com/bboylyg/NAD)
