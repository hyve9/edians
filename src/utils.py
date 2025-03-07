import csv
import re
from pypdf import PdfReader 
import tensorflow as tf

def read_labels(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        labels = {}
        label_set = set()
        for row in reader:
            if row[0] not in labels:
                labels[row[0]] = [row[1]]
            else:
                labels[row[0]].append(row[1])
            label_set.add(row[1])
    label_set.add("unknown") 
    label_mapping = {label: idx for idx, label in enumerate(label_set)}
    num_labels = len(label_set)
    
    return labels, label_mapping, num_labels
            
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        pdf_text = ''
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def extract_sections(text):
    sections = re.findall(r'(Method(?:s|ology)?|Instrument(?:s|ation)?|Measure(?:s|ments)?|Data Collection)(.*?)(Introduction|Discussion|References|Conclusion)', text, re.DOTALL)
    return ' '.join([section[1] for section in sections])

def create_dataset(papers, labels, max_len, tokenizer, batch_size, label_mapping):
    texts = list(papers.values())
    label_list = [labels[paper_id] for paper_id in papers.keys()]
    dataset = encode_samples(texts, label_list, tokenizer, max_len, label_mapping)
    return dataset.batch(batch_size)

# def cast_to_float32(data, label):
#     data = {k: tf.cast(v, tf.float32) for k, v in data.items()}
#     label = tf.cast(label, tf.float32)
#     return data, label

# def cast_to_int32(data, label):
#     data = {k: tf.cast(v, tf.int32) for k, v in data.items()}
#     label = tf.cast(label, tf.int32)
#     return data, label

def encode_samples(texts, labels, tokenizer, max_len, label_mapping):
    input_ids = []
    attention_masks = []
    label_ids = []

    for i, text in enumerate(texts):
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
        label_ids.append([label_mapping.get(label, label_mapping["unknown"]) for label in labels[i]])
    
    label_ids = tf.ragged.constant(label_ids).to_tensor(shape=(None, max_len))

    return tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tf.concat(input_ids, axis=0),
            'attention_mask': tf.concat(attention_masks, axis=0)
        },
        tf.convert_to_tensor(label_ids)
    ))