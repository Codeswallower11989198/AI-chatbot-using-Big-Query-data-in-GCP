from flask import Flask, request, jsonify
from google.cloud import bigquery
from google.cloud import aiplatform  # Import Google AI Platform
import os
from some_embedding_library import TextEmbeddingModel  # Replace with the actual library import
from vertexai.language_models import GenerativeModel  # Import Vertex AI Gemini Pro model

app = Flask(__name__)

# Initialize BigQuery client
bq_client = bigquery.Client()

# Initialize Vertex AI Platform
PROJECT_ID = os.environ.get('PROJECT_ID')
LOCATION = os.environ.get('LOCATION')
INDEX_ENDPOINT_NAME = os.environ.get('INDEX_ENDPOINT_NAME')

aiplatform.init(project=PROJECT_ID, location=LOCATION)
vertexai.init()

# Load banned phrases from environment variables
banned_phrases = os.environ.get('BANNED_QUESTIONS', '').split(",")

# Function to generate embeddings using the real TextEmbeddingModel
def generate_text_embeddings(sentences):
    """Generate text embeddings for given sentences using the TextEmbeddingModel."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")  # Load the actual model
    embeddings = model.get_embeddings([sentences])  # Generate embeddings
    vectors = [embedding.values for embedding in embeddings]  # Extract embedding values
    return vectors[0]  # Return the first embedding (assuming a single sentence)

# Function to generate context from Matching Engine results
def generate_context(matching_ids, data):
    """This function retrieves the context from a data source using the matching IDs."""
    return " ".join([data[id] for id in matching_ids])

# Function to run the Matching Engine Index Endpoint
def search_in_matching_engine(qry_emb):
    bqrelease_index_ep = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_NAME)
    response = bqrelease_index_ep.find_neighbors(
        deployed_index_id="bqrelease_index",
        queries=[qry_emb],
        num_neighbors=10
    )
    matching_ids = [neighbor.id for sublist in response for neighbor in sublist]
    return matching_ids

# Step 3: Function to check if the question contains banned phrases
def contains_banned_topics(question):
    for phrase in banned_phrases:
        if phrase.lower() in question.lower():
            return True
    return False

# Step 3: Function to augment the prompt with instructions
def augment_prompt(user_question):
    instructions = ("Please provide a friendly response. The following topics are out of context: " 
                    + ", ".join(banned_phrases) + ".")
    augmented_prompt = f"{instructions} The user asked: {user_question}"
    return augmented_prompt

# Step 4: Function to run the BigQuery ML.PREDICT query
def run_bigquery_prediction():
    """Runs a BigQuery ML.PREDICT query and returns the top 20 predictions."""
    query = """
    WITH predictions AS (
      SELECT
        *,
        ML.PREDICT(MODEL `project.dataset.your_model`, (
          SELECT
            *
          FROM
            `project.dataset.new_data`
        )) AS predicted_label
      FROM
        `project.dataset.new_data`
    )
    SELECT
      *
    FROM
      predictions
    ORDER BY
      predicted_label DESC
    LIMIT 20;
    """
    
    query_job = bq_client.query(query)
    results = query_job.result()
    
    predictions = [dict(row) for row in results]
    return predictions

# Step 5: Function to generate and send the final response back to the user
def generate_final_response(user_question, model_response):
    final_response = f"You asked: {user_question}. Here's the chatbot's response: {model_response}."
    return final_response

@app.route('/ask', methods=['POST'])
def get_user_input():
    # Extract the 'question' field from the incoming JSON request
    user_question = request.json.get('question', '')

    if user_question:
        # Step 3: Check if the question contains banned topics
        if contains_banned_topics(user_question):
            return jsonify({'response': "Sorry, your question contains inappropriate content. Please try asking something else."})
        
        # Step 3: Augment the prompt with instructions for the chatbot
        augmented_prompt = augment_prompt(user_question)
        
        # Step 2: Generate the semantic embedding for the augmented prompt
        qry_emb = generate_text_embeddings(augmented_prompt)
        
        # Step 2: Search for the best answer using the Matching Engine and context generation
        matching_ids = search_in_matching_engine(qry_emb)
        
        # Assuming you have a data source to retrieve context for matching IDs
        data = {}  # Placeholder: Replace with your actual data fetching mechanism
        context = generate_context(matching_ids, data)
        
        # Step 4: Combine the original prompt and context for Gemini Pro inference
        original_prompt = f"Based on the context delimited in backticks, answer the query, ```{context}``` {user_question}"
        full_prompt = f"{augmented_prompt} {original_prompt}"
        
        # Step 4: Use the AI model (Gemini Pro) for inference
        model = GenerativeModel("gemini-pro")
        model_response = model.predict(full_prompt).text
        
        # Step 5: Generate the final response to return to the user
        final_response = generate_final_response(user_question, model_response)
    else:
        final_response = "No question was provided. Please try again."

    # Step 5: Return the final response as JSON
    return jsonify({'response': final_response})

if __name__ == '__main__':
    app.run(debug=True)
