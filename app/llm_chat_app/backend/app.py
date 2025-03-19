import os
import json
import logging
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import openai
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Load API keys from environment variables
OPENAI_API_KEY = os.environ.get(\"OPENAI_API_KEY\")
GEMINI_API_KEY = os.environ.get(\"GEMINI_API_KEY\") # Placeholder, not used until Gemini integration

if not OPENAI_API_KEY:
    logging.error(\"OpenAI API key not found in environment variables.\")
    raise ValueError(\"OpenAI API key not found in environment variables.\")

openai.api_key = OPENAI_API_KEY

# In-memory context storage (for demonstration purposes)
conversation_history = {}  # Session ID -> list of (prompt, response) tuples

# --- LLM Abstraction Layer ---
class LLMProvider:
    def __init__(self, name):
        self.name = name

    async def generate_response(self, prompt, context, temperature, model):
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self):
        super().__init__(\"OpenAI\")

    async def generate_response(self, prompt, context, temperature, model):
        try:
            messages = [{
                \"role\": \"system\", 
                \"content\": \"You are a helpful assistant.\"
            }]
            # Include context in the messages
            for user_prompt, assistant_response in context:
                messages.append({\"role\": \"user\", \"content\": user_prompt})
                messages.append({\"role\": \"assistant\", \"content\": assistant_response})
            messages.append({\"role\": \"user\", \"content\": prompt})

            logging.info(f\"Calling OpenAI API with prompt: {prompt}, model: {model}\")

            response = await openai.ChatCompletion.acreate(
                model=model,  # Use selected model
                messages=messages,
                temperature=temperature,
                stream=True
            )
            return response
        except openai.error.RateLimitError as e:
            logging.warning(f\"OpenAI RateLimitError: {e}\")
            raise Exception(\"Rate limit exceeded. Please try again later.\") from e
        except openai.error.AuthenticationError as e:
            logging.error(f\"OpenAI AuthenticationError: {e}\")
            raise Exception(\"Invalid OpenAI API key.\") from e
        except openai.error.APIConnectionError as e:
            logging.error(f\"OpenAI APIConnectionError: {e}\")
            raise Exception(\"Failed to connect to OpenAI API.\") from e
        except openai.error.InvalidRequestError as e:
            logging.error(f\"OpenAI InvalidRequestError: {e}\")
            raise Exception(f\"Invalid request to OpenAI API: {e}\")
        except Exception as e:
            logging.exception(\"An unexpected error occurred while calling OpenAI API:\")
            raise Exception(f\"OpenAI API Error: {e}\") from e


class GeminiProvider(LLMProvider):  # Stub Implementation
    def __init__(self):
        super().__init__(\"Gemini\")

    async def generate_response(self, prompt, context, temperature, model):
        # Placeholder response
        await asyncio.sleep(1)  # Simulate some processing time
        return \"Gemini API Stub Response\"


llm_providers = {
    \"OpenAI\": OpenAIProvider(),
    \"Gemini\": GeminiProvider(),
}

# --- API Endpoints ---
@app.route(\"/chat\", methods=[\"POST\"])
async def chat():
    data = request.get_json()
    prompt = data.get(\"prompt\")
    llm_name = data.get(\"llm\", \"OpenAI\")  # Default to OpenAI
    temperature = data.get(\"temperature\", 0.7)  # Default temperature
    model = data.get(\"model\", \"gpt-3.5-turbo\") # Default model
    session_id = request.headers.get(\"Session-Id\") # Get session id from header

    if not prompt:
        return jsonify({\"error\": \"Prompt is required\"}), 400

    if llm_name not in llm_providers:
        return jsonify({\"error\": \"Invalid LLM provider\"}), 400

    if not session_id:
        return jsonify({\"error\": \"Session ID is required\"}), 400

    # Get or create conversation history for the session
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    context = conversation_history[session_id]

    full_response = \"\" # Accumulate the streamed response
    start_time = time.time()
    try:
        llm = llm_providers[llm_name]

        async def generate():
            nonlocal full_response
            try:
                response = await llm.generate_response(prompt, context, temperature, model)
                async for chunk in response:
                    if chunk and 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                      chunk_message = chunk['choices'][0]['delta']['content']
                      full_response += chunk_message  # Accumulate the response
                      yield f\"data: {json.dumps({'content': chunk_message})}\\\\\\
\\\\\\
\\\"\"
                    else:
                      #Handle edge case where the content is empty
                      pass
            except Exception as e:
                error_message = str(e)
                logging.error(f\"Error during generation: {error_message}\")
                yield f\"data: {json.dumps({'error': error_message})}\\\\\\
\\\\\\
\\\"\"
            finally:
                yield \"data: [DONE]\\\\\\
\\\\\\
\\\"\"

        response =  Response(generate(), mimetype='text/event-stream')
        logging.info(f\"Response stream started for session {session_id}\")

    except Exception as e:
        error_message = str(e)
        logging.error(f\"Error processing chat request: {error_message}\")
        return jsonify({\"error\": error_message}), 500

    finally:
        end_time = time.time()
        logging.info(f\"Full response received in {end_time - start_time:.2f} seconds\")
        # Update conversation history *after* the stream is complete
        if 'session_id' in locals():
          conversation_history[session_id].append((prompt, full_response))
          conversation_history[session_id] = conversation_history[session_id][-5:] #Keep last 5 turns
          logging.info(f\"Conversation history updated for session {session_id}\")

    return response


@app.route(\"/clear_context\", methods=[\"POST\"])
def clear_context():
    session_id = request.headers.get(\"Session-Id\")
    if session_id in conversation_history:
        del conversation_history[session_id]
        logging.info(f\"Context cleared for session {session_id}\")
        return jsonify({\"message\": \"Context cleared\"})
    else:
        return jsonify({\"message\": \"No context to clear\"})


if __name__ == \"__main__\":
    app.run(debug=True, port=5000)