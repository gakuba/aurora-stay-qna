
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import os
import re

# Hybrid retrieval: dense similarity (FAISS) fused with keyword search (BM25), plus property boost and de-duplication
def keyword_tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens for BM25 (so 'ASH-SEA' becomes ['ash', 'sea'])."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Combines two rankings of the same chunk collection and fuses them with Reciprocal Rank Fusion (RRF):

    - dense ranking: FAISS similarity on the embedding model (captures meaning; weak on codes and names)
    - keyword ranking: BM25 over the chunk text (captures exact tokens such as property codes and city names)

    Two refinements address problems observed with dense-only retrieval on these documents:
    - property boost: if the question mentions a property (code, name or city, discovered automatically from
      the property matrices), chunks naming that property get a bonus so its row outranks other properties' rows;
    - de-duplication: the property matrices are repeated in three documents, so identical chunk bodies are
      returned only once, leaving room for other relevant chunks.
    """

    def __init__(self, vectorstore, candidates: int = 20, rrf_k: int = 60, property_boost: float = 0.05):
        self.vectorstore = vectorstore
        self.docs = list(vectorstore.docstore._dict.values())
        self.bm25 = BM25Okapi([keyword_tokens(d.page_content) for d in self.docs])
        self.candidates, self.rrf_k, self.property_boost = candidates, rrf_k, property_boost

        # Property aliases discovered from the matrices, e.g. "ASH-MIA Brickell, Miami FL" -> {ash-mia, mia, brickell, miami}
        text = "\n".join(d.page_content for d in self.docs)
        self.properties = {}
        for code, name, city in set(re.findall(r"(ASH-[A-Z]{3}) ([A-Z][a-zA-Z]+), ([A-Z][a-zA-Z]+)", text)):
            self.properties[code] = {code.lower(), code.split("-")[1].lower(), name.lower(), city.lower()}

    def mentioned_properties(self, query: str) -> list[str]:
        q = set(keyword_tokens(query)) | {m.lower() for m in re.findall(r"ASH-[A-Z]{3}", query)}
        return [code for code, aliases in self.properties.items() if aliases & q]

    def retrieve(self, query: str, k: int):
        # 1. Two candidate rankings
        dense = self.vectorstore.similarity_search(query, k=self.candidates)
        scores = self.bm25.get_scores(keyword_tokens(query))
        top = sorted(range(len(self.docs)), key=lambda i: -scores[i])[:self.candidates]
        keyword = [self.docs[i] for i in top if scores[i] > 0]

        # 2. Reciprocal Rank Fusion of the two rankings
        fused = {}
        for ranking in (dense, keyword):
            for rank, d in enumerate(ranking):
                fused[d.page_content] = fused.get(d.page_content, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # 3. Property boost
        for code in self.mentioned_properties(query):
            for content in fused:
                if code in content:
                    fused[content] += self.property_boost

        # 4. Top-k with de-duplication of identical chunk bodies (the text after the "title | section" line)
        by_content = {d.page_content: d for d in dense + keyword}
        results, seen = [], set()
        for content in sorted(fused, key=lambda c: -fused[c]):
            body = content.split("\n", 1)[-1].strip()
            if body in seen:
                continue
            seen.add(body)
            results.append(by_content[content])
            if len(results) == k:
                break
        return results


# One hybrid retriever per vector store (the BM25 index is built once and reused)
_hybrid_cache = {}

def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    """Retrieve the top-k relevant chunks (hybrid dense + keyword search, as developed in the notebook)."""
    if id(vectorstore) not in _hybrid_cache:
        _hybrid_cache[id(vectorstore)] = HybridRetriever(vectorstore)
    results = _hybrid_cache[id(vectorstore)].retrieve(query, k)
    return [f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}" for r in results]


# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Answer using ONLY the Retrieved Policy Information above. Do not use outside knowledge or general hotel practice.
    2. Do not assume, infer, or invent fees, times, deadlines, limits, or procedures that are not explicitly stated in the retrieved information.
    3. Property-specific facts (fees, check-in or check-out times, parking, breakfast, pet limits) apply only to the property named in the retrieved text. If the guest asks about a property that is not named in the retrieved information, say that the retrieved documents do not cover that property and that staff should check the live property profile. Never quote another property's figures as if they applied.
    4. Include every condition, exception, approval requirement, or verification step that the retrieved information attaches to the answer (for example "subject to availability", "manager approval", "plus applicable tax").
    5. If the question has several parts, answer each part separately and in order. If the retrieved information covers only some parts, answer those and state plainly which part cannot be answered from the documents.
    6. If the retrieved information is insufficient to answer, reply: "The available policy documents do not contain enough information to answer this question." and suggest the next step for staff (check the live property profile, the reservation record, or escalate to the manager on duty).
    7. If two retrieved passages conflict, say so and present both, noting which document each comes from.
    8. Write for a hotel staff member: give the direct answer in the first sentence, then the conditions. Use plain language in 2 to 6 short sentences or a short bulleted list. Do not repeat the question or copy long passages verbatim.
    9. End with a "Sources:" line listing the document title and section heading of each retrieved passage you actually used (these appear on the first line of each passage). Do not list passages you did not use.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
