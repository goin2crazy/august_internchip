from sentence_transformers import SentenceTransformer

from sentence_transformers import util

class cfg: 
  sentence_sorting_model_preset = 'sentence-transformers/all-MiniLM-L6-v2'

class SortingModel():

  def __init__(self):
      self.model = SentenceTransformer(cfg.sentence_sorting_model_preset)

  def __call__(self, sentences):
      return self.model.encode(sentences)    

  def compare_embeddings(self, embeddings):
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(*embeddings)
    return cosine_scores.numpy().squeeze()