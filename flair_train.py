from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.visual.training_curves import Plotter


columns = {1: 'text', 2: 'lemma', 3: 'pos', 4:'morph', 5:'full_morph'}
corpus: Corpus = ColumnCorpus('data', columns,
                                train_file='/lv-ud-train211.conllu',
                                dev_file = '/lv-ud-dev211.conllu',
                                test_file='/lv-ud-test211.conllu')
label_type = 'pos' #label on which to train model

label_dict = corpus.make_label_dictionary(label_type = label_type)
print(corpus)
embedding_types = [
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

tagger = SequenceTagger(
    hidden_size = 256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type=label_type,
    use_crf=True,
)

trainer = ModelTrainer(tagger,corpus)
path = '#' #insert path where to save model
trainer.train(path,
   learning_rate=0.1,
   mini_batch_size=8,
   max_epochs=200,
   checkpoint=True,
   embeddings_storage_mode='cpu',
   write_weights=True
)

plotter = Plotter()
plotter.plot_training_curves(path+'loss.tsv')
plotter.plot_weights(path+'weights.txt')