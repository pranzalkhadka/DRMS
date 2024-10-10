from nepalikit.preprocessing import TextProcessor
from nepalikit.manage_stopwords import load_stopwords
import pandas as pd
import re
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Preprocessor:
    def text_preprocessor(self, text, processor, stopwords):
        text = processor.remove_html_tags(text)
        text = processor.remove_special_characters(text)
        text = re.sub('\n', ' ', text)
        text = re.sub(r'[\d०१२३४५६७८९]', '', text)

        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]
        return ' '.join(filtered_words)

    def transform_data(self, csv_path, stopwords_path, output_dir):
        processor = TextProcessor()
        stopwords = load_stopwords(stopwords_path)
        
        dataset = pd.read_csv(csv_path)
        dataset.rename(columns={'paras': 'data'}, inplace=True)

        dataset['data'] = dataset['data'].apply(lambda text: self.text_preprocessor(text, processor, stopwords))
        dataset.drop_duplicates(inplace=True)
        dataset.dropna(inplace=True)
        
        label_encoder = LabelEncoder()
        dataset['label'] = label_encoder.fit_transform(dataset['label'])
        num_classes = len(label_encoder.classes_)
        
        train_dataset, validation_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
        train_dataset.to_csv(os.path.join(output_dir, "preprocessed_train.csv"), index=False)
        validation_dataset.to_csv(os.path.join(output_dir, "preprocessed_val.csv"), index=False)
        
        return num_classes

if __name__ == "__main__":
    csv_path = "/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/NepaliText.csv"
    stopwords_path = '/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/NepaliStopWords'
    output_dir = "/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/dataset/Classification"
    num_classes = Preprocessor().transform_data(csv_path, stopwords_path, output_dir)
    print(f"Number of classes: {num_classes}")