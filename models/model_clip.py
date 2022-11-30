from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BertTokenizer
import sys

sys.path.append("..")
from data import processor as p
import argparse


class CLIP_PreEmbedding:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.data_processor = p.Data_Processor(args, tokenizer)
        self.model = CLIPModel.from_pretrained(
            "../pretrained_model/clip-vit-base-patch32"
        )
        self.processor = CLIPProcessor.from_pretrained(
            "../pretrained_model/clip-vit-base-patch32"
        )
        self.features = self.data_processor()

    def CLIP_Preembedding(self):
        print(type(self.features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset/FB15k-237/")
    parser.add_argument("--task_name", type=str, default="FB15k-237")
    args = parser.parse_args()
    tokenizer = BertTokenizer("../pretrained_model/bert-base-uncased/vocab.txt")
    a = CLIP_PreEmbedding(args, tokenizer)
    a.CLIP_Preembedding()
