import streamlit as st
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from streamlit import cache_resource

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

@cache_resource
def get_model():
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="mps")
    return ef


@cache_resource
def get_client():
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )
    return client

@cache_resource
def get_tokenizer():
    ef = get_model()
    tokenizer = ef.model.tokenizer
    return tokenizer

# Logo
st.image("https://avatars.mds.yandex.net/get-lpc/1527204/898ac85f-b7a0-44da-8418-03271f68273b/orig", width=200)
st.title("Milvus Hybrid Search Demo")

query = st.text_input("Enter your search query:")
search_button = st.button("Search")


def doc_text_colorization(query, docs):
    tokenizer = get_tokenizer()
    query_tokens_ids = tokenizer.encode(query, return_offsets_mapping=True)
    query_tokens = tokenizer.convert_ids_to_tokens(query_tokens_ids)
    colored_texts = []

    for doc in docs:
        ldx = 0
        landmarks = []
        encoding = tokenizer.encode_plus(doc, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])[1:-1]
        offsets = encoding["offset_mapping"][1:-1]
        for token, (start, end) in zip(tokens, offsets):
            if token in query_tokens:
                if len(landmarks) != 0 and start == landmarks[-1]:
                    landmarks[-1] = end
                else:
                    landmarks.append(start)
                    landmarks.append(end)
        close = False
        color_text = ""
        for i, c in enumerate(doc):
            if ldx == len(landmarks):
                pass
            elif i == landmarks[ldx]:
                if close is True:
                    color_text += "]"
                else:
                    color_text += ":red["
                close = not close
                ldx = ldx + 1
            color_text += c
        if close is True:
            color_text += "]"
        colored_texts.append(color_text)
    return colored_texts


def hybrid_search(query_embeddings, sparse_weight=1.0, dense_weight=1.0):
    client = get_client()
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        query_embeddings["sparse"], "sparse_vector", sparse_search_params, limit=10
    )
    dense_search_params = {"metric_type": "COSINE"}
    dense_req = AnnSearchRequest(
        query_embeddings["dense"], "dense_vector", dense_search_params, limit=10
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = client.hybrid_search(
        "ru_docs",
        reqs=[sparse_req, dense_req], ranker=rerank, limit=10, output_fields=["text", "url", "metadata"]
    )
    if len(res):
        return [{"text": hit.fields["text"], "url": hit.fields["url"], "metadata": hit.fields["metadata"]} for hit in res[0]]
    else:
        return []


# Display search results when the button is clicked
if search_button and query:
    ef = get_model()
    query_embeddings = ef([query])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Dense")
        results = hybrid_search(query_embeddings, sparse_weight=0.0, dense_weight=1.0)
        texts = [result['text'] for result in results]
        link = [result['url'] for result in results]
        metadata = [result['metadata'] for result in results]
        colored_results = doc_text_colorization(query, texts)
        for i, result in enumerate(colored_results):
            # st.title(results[i]['title'])
            uri = "https://yandex.cloud/" + link[i]
            st.page_link(page=uri, label=metadata[i]["title"])
            st.markdown(result)
            # st.markdown(
            #                 f"""
            #                 <div style="width: 400px; border: 1px solid black;">
            #                     <p>{result}</p>
            #                 </div>
            #                 """,
            #                 unsafe_allow_html=True
            #             )

    with col2:
        st.header("Sparse")
        results = hybrid_search(query_embeddings, sparse_weight=1.0, dense_weight=0.0)
        texts = [result['text'] for result in results]
        link = [result['url'] for result in results]
        metadata = [result['metadata'] for result in results]
        colored_results = doc_text_colorization(query, texts)
        for i, result in enumerate(texts):
            uri = "https://yandex.cloud/" + link[i]
            st.page_link(page=uri, label=metadata[i]["title"])
            st.markdown(
                f"""
                <div style="width: 400px; border: 1px solid black;">
                    <p>{result}</p>
                </div>    
                """,
                unsafe_allow_html=True
            )

    with col3:
        st.header("Hybrid")
        results = hybrid_search(query_embeddings, sparse_weight=0.7, dense_weight=1.0)
        texts = [result['text'] for result in results]
        link = [result['url'] for result in results]
        metadata = [result['metadata'] for result in results]
        colored_results = doc_text_colorization(query, texts)
        for i, result in enumerate(texts):
            # st.title(results[i]['title'])
            uri = "https://yandex.cloud/" + link[i]
            st.page_link(page=uri, label=metadata[i]["title"])
            st.markdown(
                f"""
                <div style="width: 400px; border: 1px solid black;">
                    <p>{result}</p>
                </div>    
                """,
                unsafe_allow_html=True
            )