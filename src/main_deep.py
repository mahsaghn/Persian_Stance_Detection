import logging
import hydra
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plot
import tensorflow as tf
from hydra.core.config_store import ConfigStore
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import dill
import math
from transformers import AutoConfig, AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from feature_extractor import PSFeatureExtractor as FeatureExtractor
from config.deep_h2c import H2CDeepConfig, FeatureExtractorConf

logging.basicConfig(filename='main_deep.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

cs = ConfigStore.instance()
cs.store(name="config", node=H2CDeepConfig)
cs.store(group="",name="baseline_h2c", node=H2CDeepConfig)
cs.store(group="features",name='features',node=FeatureExtractorConf)


def convert_data_to_one_hot(y_train):
    y_train_temp = np.zeros((y_train.shape[0], 4), dtype=np.int)
    for i, label in enumerate(y_train):
        if (label == 'Agree'):
            y_train_temp[i] = [0, 0, 0, 1]
        elif (label == 'Disagree'):
            y_train_temp[i] = [0, 0, 1, 0]
        elif (label == 'Discuss'):
            y_train_temp[i] = [0, 1, 0, 0]
        else:
            y_train_temp[i] = [1, 0, 0, 0]
    return y_train_temp


def categorical_focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = float(y_true)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss, axis=1)

    return categorical_focal_loss_fixed

def recall_m(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f1_score_m2(y_true, y_pred):
    f1_0 = f1_score(y_true, y_pred, labels=[0], average=None)
    f1_1 = f1_score(y_true, y_pred, labels=[1], average=None)
    f1_2 = f1_score(y_true, y_pred, labels=[2], average=None)
    f1_3 = f1_score(y_true, y_pred, labels=[3], average=None)
    f1_macro = (f1_0[0] + f1_1[0] + f1_2[0] + f1_3[0]) / 4
    return f1_macro


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def create_inputs_targets(x_raw, cfg, padding=True):
    parsbert_tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    LENGTH = len(x_raw)
    for i in range(LENGTH):
        input = x_raw[i]  # an array has two strings inside: headline, calim

        headline = input[0]
        claim = input[1]

        sequence = parsbert_tokenizer(headline, claim, padding='max_length', max_length=cfg.MAX_LEN, truncation=True)
        input_ids = sequence['input_ids']  # encoded tokens
        token_type_ids = sequence['token_type_ids']  # segment number for every token. 0 for headline. 1 for claim.
        attention_mask = sequence['attention_mask']

        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)

    for key in dataset_dict:
        dataset_dict[key] = tf.convert_to_tensor(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    x = tf.convert_to_tensor(x)
    return x


def create_model(cfg):
    parsbert_config = AutoConfig.from_pretrained(cfg.model_path, num_labels=4)
    parsbert_model = TFAutoModel.from_pretrained(cfg.model_path, config=parsbert_config)
    labels = list(parsbert_config.label2id.keys())
    pi = 0.01
    b = -math.log((1 - pi) / pi)
    bias_initializer = keras.initializers.Constant(value=b)

    input_ids = layers.Input(shape=(cfg.MAX_LEN,), dtype=tf.int32, name='input_ids')
    token_type_ids = layers.Input(shape=(cfg.MAX_LEN,), dtype=tf.int32, name='token_type_ids')
    attention_mask = layers.Input(shape=(cfg.MAX_LEN,), dtype=tf.int32, name='attention_mask')

    bert = parsbert_model(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    hidden_mean = tf.reduce_mean(bert, axis=1)
    x = keras.layers.Dense(units=cfg.MAX_LEN, activation='relu')(hidden_mean)
    d = keras.layers.Dropout(0.5)(x)
    x1 = keras.layers.Dense(units=16, activation='relu')(d)
    classifier = keras.layers.Dense(units=4, name='classifier', activation='softmax')(x1)
    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[classifier],
    )
    return model

def categorical_focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        y_true = float(y_true)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss, axis=1)
    return categorical_focal_loss_fixed

@hydra.main(config_path="config", config_name="config")
def main(cfg: H2CDeepConfig):
    cat_inner = dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))

    psf_extractor = FeatureExtractor(cfg=cfg.features)
    tokens_claims, tokens_headlines = psf_extractor.tokenize()
    one_hot_labels = convert_data_to_one_hot(psf_extractor.labels)

    claims = np.array(psf_extractor.clean_claims)
    claims_array = np.reshape(claims, (claims.shape[0], 1))
    headlines = np.array(psf_extractor.clean_headlines)
    print(len(psf_extractor.clean_headlines))
    headlines_array = np.reshape(headlines, (headlines.shape[0], 1))
    assert (claims_array.shape == headlines_array.shape), "The features size are not equal."
    print(claims_array.shape)

    all_features = np.concatenate((claims_array, headlines_array), axis=1)
    print(all_features.shape)

    x_train_raw, x_eval_raw, y_train_raw, y_eval_raw = train_test_split(all_features, one_hot_labels, test_size=.2,
                                                                        random_state=42)

    x_train = create_inputs_targets(x_train_raw, cfg)
    y_train = y_train_raw
    print(f"{len(x_train_raw)} training points created.")
    x_eval = create_inputs_targets(x_eval_raw, cfg)
    y_eval = y_eval_raw
    print(f"{len(x_eval_raw)} evaluation points created.")

    model = create_model(cfg)

    loss = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(cfg.lr)

    class LoggingCallback(Callback):
        def __init__(self):
            Callback.__init__(self)

        def on_epoch_end(self, epoch, logs={}):
            msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.iteritems()))
            logging.info(msg)

    callbacks_list = [
        EarlyStopping(monitor='val_accuracy', patience=5),
        ModelCheckpoint(filepath=cfg.checkpoint_path, save_best_only=True, save_weights_only=True,
                        monitor='val_accuracy'),
        LoggingCallback()
    ]
    metrics_list = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), get_f1]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)
    symmarylist = []
    model.summary(print_fn=lambda x: symmarylist.append(x))
    short_model_summary = "\n".join(symmarylist)
    logging.info('Model Summary:\n' + short_model_summary)

    history = model.fit([x_train[0], x_train[1], x_train[2]], y_train,
                        validation_data=([x_eval[0], x_eval[1], x_eval[2]], y_eval),
                        batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=1, callbacks=callbacks_list)


    plot.plot(history.history['val_accuracy'], label='test')
    plot.plot(history.history['loss'], label='train')
    plot.plot(history.history['val_loss'], label='test')
    plot.legend()
    plot.savefig('output' + '.png', bbox_inches='tight')


if __name__ == '__main__':
    main()