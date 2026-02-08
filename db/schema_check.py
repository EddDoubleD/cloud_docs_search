from pymilvus import MilvusClient


client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collections = client.list_collections()
for collection in collections:
    print("Коллекция создана:", collection)