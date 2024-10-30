import os
import tensorflow as tf
from transformers import TFAutoModelForTokenClassification
from sklearn.metrics import precision_score, recall_score, f1_score

def init_model(model_name, num_labels):
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    initializer = tf.keras.initializers.TruncatedNormal(seed=123)
    model = TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.classifier = tf.keras.layers.Dense(num_labels, kernel_initializer=initializer)
    return model

def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer='adam', loss=loss, metrics=[metric])

    model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

    return model

def eval_model(model, tokenizer, papers, labels, max_len, label_mapping):
    input_texts = list(papers.values())
    true_labels = [labels[paper_id] for paper_id in papers.keys()]

    input_ids = []
    attention_masks = []

    for text in input_texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_masks})
    predicted_labels = tf.argmax(predictions.logits, axis=-1).numpy()

    true_label_ids = []
    for label_list in true_labels:
        true_label_ids.append([label_mapping.get(label, label_mapping['unknown']) for label in label_list])

    true_label_ids = tf.ragged.constant(true_label_ids).to_tensor()

    precision = precision_score(true_label_ids, predicted_labels, average='macro', zero_division='0')
    recall = recall_score(true_label_ids, predicted_labels, average='macro', zero_division='0')
    f1 = f1_score(true_label_ids, predicted_labels, average='macro', zero_division='0')

    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true_label_ids, predicted_labels), tf.not_equal(true_label_ids, 0)), tf.int32)).numpy()
    false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(true_label_ids, predicted_labels), tf.equal(predicted_labels, 1)), tf.int32)).numpy()
    false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(true_label_ids, predicted_labels), tf.equal(true_label_ids, 1)), tf.int32)).numpy()

    return precision, recall, f1, true_positives, false_positives, false_negatives