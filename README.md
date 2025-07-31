# Simple RAG
Super Simple RAG CLI for basic question answering

## Prerequisites
Run 
`pip install -r requirements.txt`

To install required libraries

You must inject your own openAI API key into the environment
`export OPENAI_API_KEY=your-api-key`

## USAGE
This CLI breaks down the RAG process into a query flow and embedding flow.
### Embedding
To embed a corpus of text (text file), run:
python main.py --embed path_to_your_text_file.txt --embeddings path_to_your_embedded_content.json

This creates a json containing chunks of text with their corresponding LLM embeddings

### Query
To ask a question with your embeddings as the grounding context, run:

python main.py --query "Why does an airplane stall?" --embeddings pilot_handbook_aeronautical_knowledge_embeddings.json

You should see answer answer, along with the context like:

*A stall is caused by exceeding the critical angle of attack (AOA), which disrupts the smooth airflow over the wing's surface. This loss of smooth airflow leads to a rapid loss of lift, and the wing stalls. A stall can occur at any airspeed or pitch attitude, depending on the flight conditions and load factors.*

*Context used: and the repeated*
*application of load factors common to high speed stalls.*
*The load factor necessary for these maneuvers produces a*
*stress on the wings and tail structure, which does not leave*
*a reasonable margin of safety in most light aircraft.*
*The only way this stall can be induced at an airspeed  flow from the wing’s surface brought on*
*by exceeding the critical AOA. A stall can occur at any pitch*
*attitude or airspeed. Stalls are one of the most misunderstood*
*areas of aerodynamics because pilots often believe an airfoil*
*stops producing lift when it stalls. In a stall, the wing does*
*not totall reased by approximately one-half at a bank of*
*approximately 63°.*
*Stalls*
*The normal stall entered from straight-and-level flight, or an*
*unaccelerated straight climb, does not produce added load*
*factors beyond the 1 G of straight-and-level flight. As the*
*stall occurs, however, this load factor may be*
