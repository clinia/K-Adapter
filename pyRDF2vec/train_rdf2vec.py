from pyrdf2vec.graphs import KG
from pyrdf2vec import RDF2VecTransformer
import numpy as np
import json
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
import tensorboard as tb

from torch.utils.tensorboard import SummaryWriter
import numpy as np


filepath = "data/custom_taxo/clinia_triples.nt"
kg = KG(filepath)

entities = kg._entities
entities_rdf = set([ent.name for ent in entities])
len(entities)


# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 22

transformer = RDF2VecTransformer(
    Word2Vec(epochs=10, vector_size=768),
    walkers=[RandomWalker(5, None, with_reverse=False, n_jobs=4, random_state=RANDOM_STATE, md5_bytes=None)],
    verbose=2,
)

# walks = transformer.get_walks(kg, list(entities_rdf))

embeddings, literals = transformer.fit_transform(kg, list(entities_rdf))

# Create embedding dict
embedding_dict = dict()
for ent in entities_rdf:
    embedding_dict[ent] = transformer.transform(kg, entities=[ent])[0][0].tolist()

# Save embedding dict
with open("pyRDF_outputs/data/embedding_dict.json", "w") as f:
    json.dump(embedding_dict, f)


# Save run for tensorboard

# Extract embeddings and tags (label and title)
# text = [inv_mappings[k] for k in embedding_dict.keys()]
text = [k.split("/")[-1] for k in embedding_dict.keys()]
values = list(embedding_dict.values())
vectors = np.array(values)
labels = ["All"] * len(text)
metadata = list(zip(text, labels))

# Create a Tensorboard event
log_dir = "."
name = "rdflib"
writer = SummaryWriter(log_dir="{}/{}".format(log_dir, name))
writer.add_embedding(vectors, metadata, metadata_header=["title", "label"])
writer.close()
