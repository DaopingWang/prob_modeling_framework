import os
import warnings
import io
import json
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#os.chdir('/Users/wang/Documents/git/lets_learn/infclean/data')
#os.chdir('C:/git/lets_learn/infclean/data')
# os.chdir('C:/Users/wangd/git/lets_learn/infclean/data')


def negative_sampling(df, target_attr, ignore_attrs, n_per_target_attr):
    # which cities exist?
    target_attr_vals = df[target_attr].astype(str).unique()
    all_attrs = list(df)
    other_attrs = list(filter(lambda x: x != target_attr, all_attrs))
    other_attrs = list(filter(lambda x: x not in ignore_attrs, other_attrs))
    positive_samples = [[] for _ in range(len(ignore_attrs) + 1)]
    negative_samples = [[] for _ in range(len(other_attrs))]

    # for each entry, sample n negative samples with city as target
    for (i, target_val) in enumerate(target_attr_vals):
        print('.', end='')
        # all entries with this city
        target_val_df = df[df[target_attr] == target_val].reset_index()
        del target_val_df['index']
        neg_attr_vals_list = []
        for attr in other_attrs:
            pos_attr_vals = target_val_df[attr].unique()
            neg_attr_vals = pd.unique(list(filter(lambda x: x not in pos_attr_vals, df[attr])))
            if len(neg_attr_vals) == 0:
                neg_attr_vals_list.append(pos_attr_vals)
            else:
                neg_attr_vals_list.append(neg_attr_vals)
        # for _ in range(n_per_target_attr * target_val_df.shape[0]):
        for _ in range(n_per_target_attr):
            # sample negative values for other attrs
            for (j, attr) in enumerate(other_attrs):
                neg_attr_val = neg_attr_vals_list[j][np.random.randint(0, len(neg_attr_vals_list[j]))]
                negative_samples[j].append(neg_attr_val)
            positive_samples[0].append(target_val)
            # sample positive values for ignored attributes
            for (j, attr) in enumerate(ignore_attrs):
                positive_samples[j + 1].append(target_val_df[attr][np.random.randint(0, target_val_df.shape[0])])
    # [city, ignored_attrs..., other_attrs...]
    column_names = np.concatenate((np.concatenate(([target_attr], ignore_attrs)), other_attrs))
    samples = np.transpose(np.concatenate((positive_samples, negative_samples)))
    negative_samples_df = pd.DataFrame(samples, columns=column_names)
    negative_samples_df['good'] = np.zeros(negative_samples_df.shape[0])
    return negative_samples_df


print('TENSORFLOW VERSION: {}'.format(tf.__version__))
if not tf.test.gpu_device_name():
    warnings.warn('NO GPU FOUND')
else:
    print('DEFAULT GPU DEVICE: {}'.format(tf.test.gpu_device_name()))
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_w2v_single_vocab_model(cat_input_length, num_input_length, voc_size, e_dim):
    word_input_list = []
    word_input_name = 'word_input_{}'
    num_input_list = []
    num_input_name = 'num_input_{}'
    for i in range(cat_input_length):
        input_i = keras.layers.Input(shape=(1,), name=word_input_name.format(i))
        word_input_list.append(input_i)
    for i in range(num_input_length):
        numerical_input_i = keras.layers.Input(shape=(1,), name=num_input_name.format(i))
        num_input_list.append(numerical_input_i)

    embedding = keras.layers.Embedding(input_dim=voc_size,
                                       output_dim=e_dim,
                                       input_length=cat_input_length,
                                       name='embedding')
    word_reshaped_list = []
    word_reshaped_name = 'encode_reshape_{}'
    for i in range(cat_input_length):
        encoded_i = embedding(word_input_list[i])
        reshape_i = keras.layers.Reshape((e_dim,), name=word_reshaped_name.format(i))(encoded_i)
        word_reshaped_list.append(reshape_i)

    dot_sim_list = []
    dot_sim_name = 'dot_sim_{}_{}'
    for i in range(cat_input_length - 1):
        for j in range(i + 1, cat_input_length):
            dot_sim_i = keras.layers.dot([word_reshaped_list[i], word_reshaped_list[j]],
                                         axes=1,
                                         normalize=True,
                                         name=dot_sim_name.format(i, j))
            dot_sim_list.append(dot_sim_i)
    merge_sim = keras.layers.concatenate(dot_sim_list, axis=1) if len(dot_sim_list) > 1 else dot_sim_list[0]
    merge_num = keras.layers.concatenate(num_input_list, axis=1) if len(num_input_list) > 1 else num_input_list[0]
    merge_final = keras.layers.concatenate([merge_sim, merge_num], axis=1)
    output = keras.layers.Dense(units=1, activation='sigmoid')(merge_final)

    word_input_list.extend(num_input_list)
    m = keras.Model(inputs=word_input_list, outputs=output, name='cbow_model')
    # m.summary()
    m.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    keras.utils.plot_model(m, to_file='w2v_single_vocab.png')
    return m

def get_w2v_multi_vocab_model(cat_input_length, num_input_length, voc_size, e_dim):
    assert cat_input_length > 1, 'Input length must be greater than 1, current: %i' % cat_input_length
    word_input_list = []
    word_input_name = 'word_input_{}'
    num_input_list = []
    num_input_name = 'num_input_{}'
    for i in range(cat_input_length):
        input_i = keras.layers.Input(shape=(1,), name=word_input_name.format(i))
        word_input_list.append(input_i)
    for i in range(num_input_length):
        numerical_input_i = keras.layers.Input(shape=(1,), name=num_input_name.format(i))
        num_input_list.append(numerical_input_i)

    word_encode_list = []
    word_encode_name = 'encode_reshape_{}'
    for i in range(cat_input_length):
        encode_i = keras.layers.Embedding(input_dim=voc_size[i],
                                          output_dim=e_dim,
                                          input_length=1,
                                          embeddings_initializer="glorot_uniform",
                                          #embeddings_regularizer=keras.regularizers.l2(l=0.01),
                                          )(word_input_list[i])
        reshape_i = keras.layers.Reshape((e_dim,), name=word_encode_name.format(i))(encode_i)
        word_encode_list.append(reshape_i)

    dot_sim_list = []
    dot_sim_name = 'dot_sim_{}_{}'
    for i in range(cat_input_length - 1):
        for j in range(i + 1, cat_input_length):
            dot_sim_i = keras.layers.dot([word_encode_list[i], word_encode_list[j]],
                                         axes=1,
                                         normalize=False,
                                         name=dot_sim_name.format(i, j))
            dot_sim_list.append(dot_sim_i)
    merge_sim = keras.layers.concatenate(dot_sim_list, axis=1) if len(dot_sim_list) > 1 else dot_sim_list[0]
    merge_num = keras.layers.concatenate(num_input_list, axis=1) if len(num_input_list) > 1 else num_input_list[0]
    merge_final = keras.layers.concatenate([merge_sim, merge_num], axis=1)
    output = keras.layers.Dense(units=1, activation='sigmoid')(merge_final)

    word_input_list.extend(num_input_list)
    m = keras.Model(inputs=word_input_list, outputs=output, name='cbow_model')
    # m.summary()
    m.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    # keras.utils.plot_model(m, to_file='w2v_multi_vocab.png')
    return m


#############################################
CATEGORICAL_ATTR = ['catalog_id',
                #'article_id',
                #'lower_bound',
                'unit',
                'keywords',
                'manufacturer_name',
                #'ean',
                'set_id',
                    ]
NUMERICAL_ATTR = ['ek_amount',
                #'vk_amount',
                    ]
DROP_ATTR = ['negative_sample',
             'article_id',
             #'unit',
             ]

TRAIN_FILENAME = "C:\\probabilistic_modeling_framework\\data\\mercateo\\20200227_showcase\\merc_k_0.05_p_0.2.csv"
EMBEDDING_DIM = 100
batchsize = 4096

train_df = pd.read_csv(TRAIN_FILENAME)
train_df = train_df.drop(columns=DROP_ATTR)
#neg_df = negative_sampling(train_df, 'keywords', ['ek_amount'], 5)
train_df['good'] = np.ones(train_df.shape[0])
#train_df = pd.concat([train_df, neg_df], ignore_index=True)
# shuffle
train_df = train_df.sample(frac=1).reset_index(drop=True)

# Load dataset into separate vocabularies, integer-encoded
unique_values = [train_df[cat_attr].astype(str).unique() for cat_attr in CATEGORICAL_ATTR]
keys = [pd.Categorical(u).codes for u in unique_values]
category_dict = []
for i, key_set in enumerate(keys):
    category_dict.append(dict(zip(key_set, unique_values[i])))
    train_df[CATEGORICAL_ATTR[i]] = train_df[CATEGORICAL_ATTR[i]].astype(str).replace(unique_values[i], key_set)

# multi vocab
vocabulary_size = [len(u) for u in unique_values]
# adaptive_embedding_dim = min(EMBEDDING_DIM, int(sum(vocabulary_size)**0.25))
adaptive_embedding_dim = EMBEDDING_DIM
model = get_w2v_multi_vocab_model(len(CATEGORICAL_ATTR), len(NUMERICAL_ATTR), vocabulary_size, adaptive_embedding_dim)

# Generate labels, i.e. we label all observations as positive samples
# train_target = np.ones(train_df.shape[0])
train_target = train_df['good'].to_numpy().astype(int)

train_np = [train_df[cat_attr].to_numpy().astype(int) for cat_attr in CATEGORICAL_ATTR]
for num_attr in NUMERICAL_ATTR:
    train_np.append(train_df[num_attr].to_numpy().astype(float))

t = time.time()
history = model.fit(x=train_np,
                    y=train_target,
                    batch_size=batchsize,
                    epochs=2000)

elapsed = time.time() - t
print(elapsed)

# Vector IO
# Save embeddings from one vocab
'''e = model.layers[3]
weights = e.get_weights()[0]
out_v = io.open('w2v_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('w2v_meta.tsv', 'w', encoding='utf-8')

for i in range(0, vocabulary_size):
    vec = weights[i]
    out_m.write(category_dict[i] + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()'''

# Save for multiple vocabs
tsv_name = 'ek_100_4069_2000{}_{}.tsv'
for i, cat_attr in enumerate(CATEGORICAL_ATTR, start=0):
    e = model.layers[len(CATEGORICAL_ATTR) + i]
    weights = e.get_weights()[0]
    out_v = io.open(tsv_name.format(cat_attr, 'vec'), 'w', encoding='utf-8')
    out_m = io.open(tsv_name.format(cat_attr, 'meta'), 'w', encoding='utf-8')

    for j in range(0, vocabulary_size[i]):
        vec = weights[j]
        out_m.write(category_dict[i][j] + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()


print('SUCCESS')