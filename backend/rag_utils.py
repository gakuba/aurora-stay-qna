
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

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
