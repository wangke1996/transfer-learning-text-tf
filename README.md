# Transfer Learning for Text Classification with Tensorflow, fork fron
# forked from dongjun-Lee/transfer-learning-text-tf

Tensorflow implementation of [Semi-supervised Sequence Learning(https://arxiv.org/abs/1511.01432)](https://arxiv.org/abs/1511.01432).
 
Auto-encoder or language model is used as a pre-trained model to initialize LSTM text classification model.

- ***SA-LSTM***: Use auto-encoder as a pre-trained model.
- ***LM-LSTM***: Use language model as a pre-trained model.


## Requirements
- Python 3
- Tensorflow
- pip install -r requirements.txt

## Usage
DBpedia dataset is used for pre-training and training.

### Pre-train auto encoder or language model
```
$ python pre_train.py --model="<MODEL>" --model_name="<MODEL_NAME>" --dict_size="<DICT_SIZE>"
```
(\<Model>: auto_encoder | language_model)
(\<Model_Name>: Give a name to the model, default to "model")
(\<Dict_Szie>: The limit of vocabulary dictionary size, default to 20000)
### Train LSTM text classification model
```
$ python train.py --pre_trained="<MODEL>" --model_name="<MODEL_NAME>"
```
(\<Model>: none | auto_encoder | language_model)
(\<Model_Name>: The pretrained model's name, default to "model")

