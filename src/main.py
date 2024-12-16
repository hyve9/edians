import sys
import logging
import argparse
import pathlib
import utils as u
import model as m
from transformers import LongformerTokenizer

if __name__ == "__main__":

    if sys.version_info.major < 3:
        logging.error('Please use Python 3.x or higher')
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='CSV containing labels')
    parser.add_argument('--data', type=str, required=True, help='Data directory (currently supports PDFs only)')
    parser.add_argument('--model-checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--tokenizer-checkpoint', type=str, help='Path to tokenizer checkpoint')
    parser.add_argument('--save-checkpoint', action='store_true', help='Save model checkpoint')
    parser.add_argument('--log', type=str, default='warn', help='Log level (choose from debug, info, warning, error and critical)')

    args = parser.parse_args()

    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    loglevel = args.log.lower() if (args.log.lower() in levels) else 'warn'
    logging.basicConfig(stream=sys.stderr, level=levels[loglevel], format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    max_len = 1024
    num_epochs = 3
    batch_size = 16
    learning_rate = 3e-5
    model_name = 'allenai/longformer-base-4096'

    if args.tokenizer_checkpoint:
        tokenizer = LongformerTokenizer.from_pretrained(args.tokenizer_checkpoint)
    else:
        tokenizer = LongformerTokenizer.from_pretrained(model_name)

    papers = {}

    # Get labels
    labels, label_mapping, num_labels = u.read_labels(args.labels)
    data_dir = pathlib.Path(args.data)
    if not data_dir.exists():
        logging.error(f'{data_dir} does not exist')
        sys.exit(1)

    # Process PDFs
    for pdf in data_dir.glob('*.pdf'):
        pdf_name = pdf.stem
        pdf_text = u.process_pdf(pdf)
        sections = u.extract_sections(pdf_text)
        papers[pdf_name] = sections

    # Create dataset
    dataset = u.create_dataset(papers, labels, max_len, tokenizer, batch_size, label_mapping)
    train_size = int(0.8 * len(list(dataset)))
    train_dataset, val_dataset = dataset.take(train_size), dataset.skip(train_size)
    
    # Initialize model
    if args.model_checkpoint:
        model = m.init_model(pathlib.Path(args.model_checkpoint), num_labels)
    else:
        model = m.init_model(model_name, num_labels)

    # Train model
    model = m.train_model(model, train_dataset, val_dataset, num_epochs, learning_rate)

    # Evaluate model
    precision, recall, f1_score, true_positives, false_positives, false_negatives = m.eval_model(model, tokenizer, papers, labels, max_len, label_mapping)

    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F1 Score: {f1_score}')
    logging.info(f'True Positives: {true_positives}')
    logging.info(f'False Positives: {false_positives}')
    logging.info(f'False Negatives: {false_negatives}')

    if args.save_checkpoint:
        model_checkpoint = f'edians-model-{num_epochs}-{batch_size}-{max_len}-{precision:.2f}-{recall:.2f}-{f1_score:.2f}'
        tokenizer_checkpoint = f'edians-tokenizer-{num_epochs}-{batch_size}-{max_len}-{precision:.2f}-{recall:.2f}-{f1_score:.2f}'
        model.save_pretrained(f'checkpoints/{model_checkpoint}')
        tokenizer.save_pretrained(f'checkpoints/{tokenizer_checkpoint}')
