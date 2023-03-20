# CS6910 - Fundamentals of Deep Learning: Assignment-1

By Dhananjay Balakrishnan, ME19B012.
Submitted as part of the course CS6910, taught by Professor Mitesh Khapra, Jan-May 2023 semester.

The WandB report corresponding to this code can be found here: https://api.wandb.ai/links/clroymustang/19pi47oa.

## Instructions to use:
### Training the Neural Network.
0. Install the requirements from the 'requirements.txt' file.
```
pip install -r requirements.txt
```
1. To train the Neural Network, use the 'train.py' script. Here is the format of how you should run it:

```
python train.py <arguments>
```

The Compulsory Arguments are:
|Argument Name|Description|Default Value|
| ------------- | ------------- | -------- |
|-wp, --wandb_project|WandB Project Name|''|
|-we, --wandb_entity|WandB Entity Name|''|

The optional arguments are:
|Argument Name|Description|Default Value|
| ------------- | ------------- | -------- |
|-d, --dataset|Dataset type. Either 'mnist' or 'fashion_mnist'.|'fashion_mnist'|
|-e, --epochs|Number of Epochs|10|
|-b, --batch_size|Batch Size|32|
|-l, --loss|Loss Function. Either 'cross_entropy' or 'MSE'.|'cross_entropy'|
|-o, --optimizer|Optimizer. Choose one from ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'].|'nadam'|
|-lr, --learning_rate|Learning Rate|0.001|
|-m, --momentum|Momentum|0.9|
|-beta, --beta|beta for rmsprop|0.9|
|-beta1, --beta1|beta1 for adam and nadam.|0.9|
|-beta2, --beta2|beta2 for adam and nadam.|0.99|
|-eps, --epsilon|Epsilon used by various optimizers.|0.000001|
|-w_d, --weight_decay|Weight Decay, for L2 Regularization|0.0005|
|-w_i, --weight_init|Mode of Initialization. Either 'random' or 'Xavier'.|'Xavier'|
|-nhl, --num_layers|Number of Hidden Layers|3|
|-sz, --hidden_size|Hidden Layer Size|128|
|-a, --activation|Activation Function. Choose one from ['sigmoid', 'tanh', 'identity', 'ReLU'].|'tanh'|
|-rn, --run_name|Name of WandB run.|'model_run'|

2. On completion, the model will be stored in the same directory as 'model.pkl'. You can save the model to recreate the results later. 

## Assignment Specific Functions and WandB sweep
The WandB hyperparameter sweep, and all required comparisons for the various questions of the problem statement have been performed in the Jupyter Notebook titled 'assignment1_nb.ipynb'.

##Best-Performing Models
The models that performed the best on the Fashion-MNIST dataset can be found as pickled objects in the 'best_models' folder.
