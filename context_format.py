import pinecone
from langchain_community.embeddings import OpenAIEmbeddings


# Embed and query Pinecone
def retrieve_context(index, query, openai_client, cache, top_k=5):
    # Return cached results if available
    if query in cache:
        return cache[query]

    # Generate embedding for query using OpenAI embeddings
    embed_model = OpenAIEmbeddings(openai_api_key=openai_client.api_key)
    query_embedding = embed_model.embed_query(query)

    # Query Pinecone index with embedding
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

    # Extract matches
    matches = results.matches if hasattr(results, 'matches') else []

    # Cache and return
    cache[query] = matches
    return matches

def format_context(matches, max_chunks=5):
    """
    Format retrieved context chunks into a smooth, immersive storytelling style.

    Args:
        matches (list): List of retrieval matches with metadata.
        max_chunks (int): Max number of chunks to include for context.

    Returns:
        str: Formatted context string with narrative flow and markdown styling.
    """
    formatted = []
    for i, match in enumerate(matches[:max_chunks]):
        chunk = match.metadata.get("text", "") or match.metadata.get("page_content", "")
        title = match.metadata.get("video_title", "Unknown Title")
        source_url = match.metadata.get("source_url", None)  # optional source link
        
        # Add smooth transitions for storytelling flow
        if i == 0:
            intro = f"Let's begin our immersive story inspired by **'{title}'**:\n"
        else:
            intro = f"\nContinuing with insights from **'{title}'**:\n"
        
        # Clean chunk text
        chunk_text = chunk.strip()
        
        # Optionally add source link if available
        source_line = f"\n[Watch the video here]({source_url})" if source_url else ""
        
        # Append formatted chunk with transitions and optional source
        formatted.append(intro + chunk_text + source_line)
    
    # Join all chunks with horizontal rule separator for clarity
    return "\n\n---\n\n".join(formatted)


# Example usage (you would integrate this in your story generation workflow)

def example_story_generation(index, openai_client, cache, user_question):
    # Retrieve relevant context chunks for user's query
    matches = retrieve_context(index, user_question, openai_client, cache, top_k=5)
    
    # Format retrieved chunks into immersive storytelling context
    context = format_context(matches)
    
    # Now you can pass `context` into your LLM prompt for story generation
    print("Formatted Context for Storytelling:\n")
    print(context)