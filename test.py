from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import argparse

class KEYBERT():
    def __init__(self):

        self.sentence_model = SentenceTransformer('/sentence-transformer')
        self.kw_model = KeyBERT(model=self.sentence_model)

    def extract(self, doc):
        print(self.kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None))
        print(self.kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None))
        


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Sentence Transformer model + KeyBert.')
  parser.add_argument('--model',default=None, help='Sentence Transformer path')

  args = parser.parse_args()
  model=KEYBERT(args)
  model.extract(doc=' المناخ مداري توجد البلاد فصول رئيسية  ربيع صيف ممطر شتاء معتدل فدرجات الحرارة تصل المرتفعات الجبلية مئوية جانب هبوب الرياح الموسمية الغربية الجنوبي')
