#######################
# >
#
#
#######################
# BGE-M3 model can embed texts as dense and sparse vectors.
# It is included in the optional `model` module in pymilvus, to install it,
# simply run "pip install 'pymilvus[model]'".
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from pymilvus import (
    DataType, MilvusClient,
)

ef = BGEM3EmbeddingFunction(use_fp16=False, device="mps")
dense_dim = ef.dim["dense"]
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=False,
)

schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1024)
schema.add_field(field_name="metadata", datatype=DataType.JSON)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)

# automatically decides the most appropriate index type
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP"
)

# create collection
client.create_collection(
    collection_name="ru_docs",
    schema=schema,
    index_params=index_params
)
