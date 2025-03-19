**Here you can check all the code explanation.**


Okay, I understand. I'll provide a comprehensive explanation of the provided code, covering each file and block in detail.

# LLM Chat Application

This is a simple chat application that uses a Large Language Model (LLM) to generate responses to user prompts. It consists of a frontend built with React and a backend built with Flask.

## Project Structure

```
llm_chat_app/
├── backend/
│   ├── app.py
│   ├── tests/
│   │   └── test_app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── index.css
│   │   ├── setupTests.js
│   ├── public/
│   │   └── index.html
│   ├── Dockerfile
│   ├── package.json
│   └── README.md
├── docker-compose.yml
└── README.md
```

## Backend (`backend/`)

The backend is responsible for handling API requests, interacting with the LLM, and managing conversation history.

### `backend/app.py`

This is the main application file for the backend. It defines the Flask application, API endpoints, and logic for interacting with the LLM.

```python
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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Placeholder, not used until Gemini integration

if not OPENAI_API_KEY:
    logging.error("OpenAI API key not found in environment variables.")
    raise ValueError("OpenAI API key not found in environment variables.")

openai.api_key = OPENAI_API_KEY

# In-memory context storage (for demonstration purposes)
conversation_history = {}  # Session ID -> list of (prompt, response) tuples

# --- LLM Abstraction Layer ---
class LLMProvider:
    def __init__(self, name):
        self.name = name

    async def generate_response(self, prompt, context, temperature):
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self):
        super().__init__("OpenAI")

    async def generate_response(self, prompt, context, temperature):
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            # Include context in the messages
            for user_prompt, assistant_response in context:
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": assistant_response})
            messages.append({"role": "user", "content": prompt})

            logging.info(f"Calling OpenAI API with prompt: {prompt}")

            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",  # Or another model you prefer
                messages=messages,
                temperature=temperature,
                stream=True
            )
            return response
        except openai.error.RateLimitError as e:
            logging.warning(f"OpenAI RateLimitError: {e}")
            raise Exception("Rate limit exceeded. Please try again later.") from e
        except openai.error.AuthenticationError as e:
            logging.error(f"OpenAI AuthenticationError: {e}")
            raise Exception("Invalid OpenAI API key.") from e
        except openai.error.APIConnectionError as e:
            logging.error(f"OpenAI APIConnectionError: {e}")
            raise Exception("Failed to connect to OpenAI API.") from e
        except openai.error.InvalidRequestError as e:
            logging.error(f"OpenAI InvalidRequestError: {e}")
            raise Exception(f"Invalid request to OpenAI API: {e}")
        except Exception as e:
            logging.exception("An unexpected error occurred while calling OpenAI API:")
            raise Exception(f"OpenAI API Error: {e}") from e


class GeminiProvider(LLMProvider):  # Stub Implementation
    def __init__(self):
        super().__init__("Gemini")

    async def generate_response(self, prompt, context, temperature):
        # Placeholder response
        await asyncio.sleep(1)  # Simulate some processing time
        return "Gemini API Stub Response"


llm_providers = {
    "OpenAI": OpenAIProvider(),
    "Gemini": GeminiProvider(),
}

# --- API Endpoints ---
@app.route("/chat", methods=["POST"])
async def chat():
    data = request.get_json()
    prompt = data.get("prompt")
    llm_name = data.get("llm", "OpenAI")  # Default to OpenAI
    temperature = data.get("temperature", 0.7)  # Default temperature
    session_id = request.headers.get("Session-Id") # Get session id from header

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if llm_name not in llm_providers:
        return jsonify({"error": "Invalid LLM provider"}), 400

    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    # Get or create conversation history for the session
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    context = conversation_history[session_id]

    full_response = "" # Accumulate the streamed response
    start_time = time.time()
    try:
        llm = llm_providers[llm_name]

        async def generate():
            nonlocal full_response
            try:
                response = await llm.generate_response(prompt, context, temperature)
                async for chunk in response:
                    if chunk and 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                      chunk_message = chunk['choices'][0]['delta']['content']
                      full_response += chunk_message  # Accumulate the response
                      yield f"data: {json.dumps({'content': chunk_message})}\n\n"
                    else:
                      #Handle edge case where the content is empty
                      pass
            except Exception as e:
                error_message = str(e)
                logging.error(f"Error during generation: {error_message}")
                yield f"data: {json.dumps({'error': error_message})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        response =  Response(generate(), mimetype='text/event-stream')
        logging.info(f"Response stream started for session {session_id}")

    except Exception as e:
        error_message = str(e)
        logging.error(f"Error processing chat request: {error_message}")
        return jsonify({"error": error_message}), 500

    finally:
        end_time = time.time()
        logging.info(f"Full response received in {end_time - start_time:.2f} seconds")
        # Update conversation history *after* the stream is complete
        if 'session_id' in locals():
          conversation_history[session_id].append((prompt, full_response))
          conversation_history[session_id] = conversation_history[session_id][-5:] #Keep last 5 turns
          logging.info(f"Conversation history updated for session {session_id}")

    return response


@app.route("/clear_context", methods=["POST"])
def clear_context():
    session_id = request.headers.get("Session-Id")
    if session_id in conversation_history:
        del conversation_history[session_id]
        logging.info(f"Context cleared for session {session_id}")
        return jsonify({"message": "Context cleared"})
    else:
        return jsonify({"message": "No context to clear"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

#### Imports

```python
import os
import json
import logging
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import openai
import asyncio
```

*   `os`: For interacting with the operating system (e.g., accessing environment variables).
*   `json`: For encoding and decoding JSON data.
*   `logging`: For logging events and errors.  Good logging is *essential* for debugging and monitoring applications in production.
*   `time`: For measuring execution time.
*   `flask`:  The core Flask library for creating the web application.  `Flask` provides the basic framework for defining routes, handling requests, and generating responses.
*   `request`:  Flask's request object, used to access incoming request data (e.g., headers, JSON payload).
*   `jsonify`:  Flask's helper function for creating JSON responses.
*   `Response`: Flask's class to handle complex responses, in this case, stream responses.
*   `flask_cors`: For handling Cross-Origin Resource Sharing (CORS).  CORS is *crucial* for allowing the frontend (running on a different domain/port) to make requests to the backend.
*   `openai`:  The OpenAI Python library for interacting with the OpenAI API.
*   `asyncio`: For asynchronous programming.  This is *important* for handling I/O-bound operations (like API calls) without blocking the main thread.  Using `asyncio` allows the server to handle more requests concurrently.

#### Logging Configuration

```python
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

This configures the logging system to output messages at the `INFO` level or higher (e.g., `WARNING`, `ERROR`, `CRITICAL`). The format string specifies the structure of the log messages, including the timestamp, log level, and message.

#### Flask App Initialization

```python
app = Flask(__name__)
CORS(app)
```

*   `app = Flask(__name__)`: Creates a Flask application instance. `__name__` is a special Python variable that represents the name of the current module.
*   `CORS(app)`: Enables CORS for the entire application. This allows requests from any origin.  In a production environment, you would typically want to restrict the allowed origins for security reasons.

#### API Key Loading

```python
# Load API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Placeholder, not used until Gemini integration

if not OPENAI_API_KEY:
    logging.error("OpenAI API key not found in environment variables.")
    raise ValueError("OpenAI API key not found in environment variables.")

openai.api_key = OPENAI_API_KEY
```

*   This section retrieves the OpenAI API key from the environment variable `OPENAI_API_KEY`. **Important:** You *must* set this environment variable before running the application.
*   It also includes a placeholder for a Gemini API key.
*   If the `OPENAI_API_KEY` is not found, it logs an error and raises a `ValueError`, preventing the application from starting without a valid API key.  This is a good practice to ensure that the application doesn't try to make API calls without proper authentication.
*   `openai.api_key = OPENAI_API_KEY`: Configures the OpenAI library to use the retrieved API key.

#### Conversation History

```python
# In-memory context storage (for demonstration purposes)
conversation_history = {}  # Session ID -> list of (prompt, response) tuples
```

This initializes an in-memory dictionary to store the conversation history for each session.  The keys are session IDs, and the values are lists of (prompt, response) tuples.  **Caveat:** This is suitable for demonstration purposes only. In a production environment, you would want to use a persistent storage mechanism (e.g., a database) to store the conversation history.  If the server restarts, the in-memory history will be lost.

#### LLM Abstraction Layer

```python
# --- LLM Abstraction Layer ---
class LLMProvider:
    def __init__(self, name):
        self.name = name

    async def generate_response(self, prompt, context, temperature):
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self):
        super().__init__("OpenAI")

    async def generate_response(self, prompt, context, temperature):
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            # Include context in the messages
            for user_prompt, assistant_response in context:
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": assistant_response})
            messages.append({"role": "user", "content": prompt})

            logging.info(f"Calling OpenAI API with prompt: {prompt}")

            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",  # Or another model you prefer
                messages=messages,
                temperature=temperature,
                stream=True
            )
            return response
        except openai.error.RateLimitError as e:
            logging.warning(f"OpenAI RateLimitError: {e}")
            raise Exception("Rate limit exceeded. Please try again later.") from e
        except openai.error.AuthenticationError as e:
            logging.error(f"OpenAI AuthenticationError: {e}")
            raise Exception("Invalid OpenAI API key.") from e
        except openai.error.APIConnectionError as e:
            logging.error(f"OpenAI APIConnectionError: {e}")
            raise Exception("Failed to connect to OpenAI API.") from e
        except openai.error.InvalidRequestError as e:
            logging.error(f"OpenAI InvalidRequestError: {e}")
            raise Exception(f"Invalid request to OpenAI API: {e}")
        except Exception as e:
            logging.exception("An unexpected error occurred while calling OpenAI API:")
            raise Exception(f"OpenAI API Error: {e}") from e


class GeminiProvider(LLMProvider):  # Stub Implementation
    def __init__(self):
        super().__init__("Gemini")

    async def generate_response(self, prompt, context, temperature):
        # Placeholder response
        await asyncio.sleep(1)  # Simulate some processing time
        return "Gemini API Stub Response"


llm_providers = {
    "OpenAI": OpenAIProvider(),
    "Gemini": GeminiProvider(),
}
```

This section defines an abstraction layer for interacting with different LLMs.

*   `LLMProvider`:  A base class for LLM providers. It defines the `generate_response` method, which is responsible for generating a response to a prompt.  This class uses `NotImplementedError`, ensuring that subclasses implement this method.  This promotes code organization and maintainability, making it easier to add or switch between different LLMs in the future.
*   `OpenAIProvider`: A subclass of `LLMProvider` that implements the `generate_response` method for the OpenAI API.
    *   It constructs a list of messages, including a system message ("You are a helpful assistant.") and the conversation history. The conversation history is formatted as a series of user and assistant messages.
    *   It calls the `openai.ChatCompletion.acreate` method to generate a response.
    *   It handles potential OpenAI API errors, such as rate limits, authentication errors, and API connection errors.  The error handling is robust, logging the specific error and raising a more user-friendly exception.
    *   The `stream=True` argument enables streaming responses from the OpenAI API.
*   `GeminiProvider`: A placeholder implementation for the Gemini API.  Currently, it simply returns a stub response.
*   `llm_providers`: A dictionary that maps LLM names to their corresponding provider instances. This allows the application to easily switch between different LLMs.

#### `/chat` Endpoint

```python
# --- API Endpoints ---
@app.route("/chat", methods=["POST"])
async def chat():
    data = request.get_json()
    prompt = data.get("prompt")
    llm_name = data.get("llm", "OpenAI")  # Default to OpenAI
    temperature = data.get("temperature", 0.7)  # Default temperature
    session_id = request.headers.get("Session-Id") # Get session id from header

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if llm_name not in llm_providers:
        return jsonify({"error": "Invalid LLM provider"}), 400

    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    # Get or create conversation history for the session
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    context = conversation_history[session_id]

    full_response = "" # Accumulate the streamed response
    start_time = time.time()
    try:
        llm = llm_providers[llm_name]

        async def generate():
            nonlocal full_response
            try:
                response = await llm.generate_response(prompt, context, temperature)
                async for chunk in response:
                    if chunk and 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                      chunk_message = chunk['choices'][0]['delta']['content']
                      full_response += chunk_message  # Accumulate the response
                      yield f"data: {json.dumps({'content': chunk_message})}\n\n"
                    else:
                      #Handle edge case where the content is empty
                      pass
            except Exception as e:
                error_message = str(e)
                logging.error(f"Error during generation: {error_message}")
                yield f"data: {json.dumps({'error': error_message})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        response =  Response(generate(), mimetype='text/event-stream')
        logging.info(f"Response stream started for session {session_id}")

    except Exception as e:
        error_message = str(e)
        logging.error(f"Error processing chat request: {error_message}")
        return jsonify({"error": error_message}), 500

    finally:
        end_time = time.time()
        logging.info(f"Full response received in {end_time - start_time:.2f} seconds")
        # Update conversation history *after* the stream is complete
        if 'session_id' in locals():
          conversation_history[session_id].append((prompt, full_response))
          conversation_history[session_id] = conversation_history[session_id][-5:] #Keep last 5 turns
          logging.info(f"Conversation history updated for session {session_id}")

    return response
```

This endpoint handles incoming chat requests.

*   `@app.route("/chat", methods=["POST"])`: Defines the route for the chat endpoint. It only accepts POST requests.
*   `data = request.get_json()`: Parses the JSON payload from the request body.
*   `prompt = data.get("prompt")`: Extracts the prompt from the JSON data.
*   `llm_name = data.get("llm", "OpenAI")`: Extracts the LLM name from the JSON data. It defaults to "OpenAI" if no LLM name is provided.
*   `temperature = data.get("temperature", 0.7)`: Extracts the temperature from the JSON data. It defaults to 0.7 if no temperature is provided.  The temperature parameter controls the randomness of the generated responses.  A higher temperature will result in more random responses, while a lower temperature will result in more deterministic responses.
*   `session_id = request.headers.get("Session-Id")`: Retrieves the session ID from the `Session-Id` header.  Using headers for passing the session ID is a reasonable choice, although cookies are another common option.
*   The code then validates the input, ensuring that the prompt, LLM name, and session ID are provided.  If any of these are missing or invalid, it returns a 400 error with a JSON payload indicating the error.  Input validation is *critical* for preventing errors and security vulnerabilities.
*   `if session_id not in conversation_history: conversation_history[session_id] = []`:  Retrieves or creates the conversation history for the session.
*   `context = conversation_history[session_id]`: Gets the conversation history for the current session.
*   The code then calls the selected LLM to generate a response:
    *   `llm = llm_providers[llm_name]`: Gets the LLM provider instance based on the `llm_name`.
    *   `response = await llm.generate_response(prompt, context, temperature)`: Calls the `generate_response` method of the selected LLM provider.
*   The code then streams the response back to the client using Server-Sent Events (SSE):
    *  The code defines an inner asynchronous generator function `generate()`:
        *   `nonlocal full_response`: Allows the inner function to modify the `full_response` variable in the outer scope.
        *   `response = await llm.generate_response(prompt, context, temperature)`: Calls the `generate_response` method of the selected LLM provider.
        *   `async for chunk in response:`: Iterates over the chunks of the streamed response.
        *   The code checks for `chunk`, `'choices' in chunk`, `len(chunk['choices']) > 0`, `'delta' in chunk['choices'][0]` and `'content' in chunk['choices'][0]['delta']` to ensure the structure is as expected before extracting data.
        *   `chunk_message = chunk['choices'][0]['delta']['content']`: Extracts the content of the chunk.
        *   `full_response += chunk_message`: Appends the chunk to the `full_response`.
        *   `yield f"data: {json.dumps({'content': chunk_message})}\n\n"`: Yields the chunk as an SSE event.  SSE is a simple protocol for pushing real-time updates from the server to the client.
        *   The `finally` block ensures that a `[DONE]` event is sent to the client when the stream is complete, regardless of whether an error occurred.
        *   Error handling included to catch generic exceptions and stream error messages to the client to notify about backend errors.
    *   `response = Response(generate(), mimetype='text/event-stream')`: Creates a Flask `Response` object with the `generate` function as the generator and sets the `mimetype` to `text/event-stream`.
*   The `try...except...finally` block ensures that the conversation history is updated even if an error occurs during the response generation.
*   `conversation_history[session_id].append((prompt, full_response))`: Updates the conversation history with the new prompt and response.
*   `conversation_history[session_id] = conversation_history[session_id][-5:]`: Limits the conversation history to the last 5 turns.  This helps to prevent the context from growing too large and exceeding the LLM's context window.
*   The endpoint also includes timing information, logging the time it takes to generate the full response.

#### `/clear_context` Endpoint

```python
@app.route("/clear_context", methods=["POST"])
def clear_context():
    session_id = request.headers.get("Session-Id")
    if session_id in conversation_history:
        del conversation_history[session_id]
        logging.info(f"Context cleared for session {session_id}")
        return jsonify({"message": "Context cleared"})
    else:
        return jsonify({"message": "No context to clear"})
```

This endpoint clears the conversation history for a given session.

*   `@app.route("/clear_context", methods=["POST"])`: Defines the route for the clear context endpoint. It only accepts POST requests.
*   `session_id = request.headers.get("Session-Id")`: Retrieves the session ID from the `Session-Id` header.
*   `if session_id in conversation_history:`: Checks if the session ID exists in the conversation history.
*   `del conversation_history[session_id]`: Deletes the conversation history for the session.
*   It returns a JSON response indicating whether the context was cleared.

#### Main Block

```python
if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

This block starts the Flask development server when the script is executed directly.  `debug=True` enables the Flask debugger, which provides helpful error messages and allows you to reload the server automatically when you make changes to the code. **Important:** You should disable debug mode in a production environment.

### `backend/tests/test_app.py`

This file contains unit tests for the backend application.

```python
import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from backend.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_openai_response():
    # Simulate a streaming response with multiple chunks
    def generate_chunks():
        yield {'choices': [{'delta': {'content': 'Mocked '}}]}
        yield {'choices': [{'delta': {'content': 'OpenAI '}}]}
        yield {'choices': [{'delta': {'content': 'response'}}]}

    return generate_chunks()


@pytest.mark.asyncio
async def test_chat_endpoint_no_prompt(client):
    response = client.post('/chat', json={}, headers={'Session-Id': 'test_session'})
    assert response.status_code == 400
    data = json.loads(response.data.decode('utf-8'))
    assert data['error'] == "Prompt is required"


@pytest.mark.asyncio
async def test_chat_endpoint_openai(client, mocker, mock_openai_response):
    mocker.patch('backend.app.openai.ChatCompletion.acreate', new_callable=AsyncMock, return_value=mock_openai_response)

    response = client.post(
        '/chat',
        json={'prompt': 'Test prompt', 'llm': 'OpenAI'},
        headers={'Session-Id': 'test_session'}
    )

    assert response.status_code == 200
    # Simulate reading the stream
    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                data = json.loads(decoded_line[5:])
                if 'content' in data:
                    full_response += data['content']

    assert "Mocked OpenAI response" in full_response



@pytest.mark.asyncio
async def test_chat_endpoint_gemini_stub(client):
    response = client.post(
        '/chat',
        json={'prompt': 'Test prompt', 'llm': 'Gemini'},
        headers={'Session-Id': 'test_session'}
    )
    assert response.status_code == 200
    # Simulate reading the stream
    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                data = json.loads(decoded_line[5:])
                if 'content' in data:
                    full_response += data['content']
    assert "Gemini API Stub Response" in full_response


@pytest.mark.asyncio
async def test_clear_context(client):
    # First, create some context
    client.post(
        '/chat',
        json={'prompt': 'Test prompt', 'llm': 'OpenAI'},
        headers={'Session-Id': 'test_session'}
    )
    response = client.post('/clear_context', headers={'Session-Id': 'test_session'})
    assert response.status_code == 200
    data = json.loads(response.data.decode('utf-8'))
    assert data['message'] == "Context cleared"
```

#### Imports

```python
import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from backend.app import app
```

*   `pytest`:  The pytest testing framework.  pytest is a powerful and flexible testing framework that simplifies writing and running tests.
*   `json`: For working with JSON data.
*   `unittest.mock`: For mocking objects and functions during testing.  Mocking allows you to isolate the code being tested from external dependencies.
*   `AsyncMock`: Used for mocking async functions.
*   `MagicMock`: Used for general mocking purposes.
*   `backend.app`: Imports the Flask app instance from `backend/app.py`.

#### Fixtures

```python
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_openai_response():
    # Simulate a streaming response with multiple chunks
    def generate_chunks():
        yield {'choices': [{'delta': {'content': 'Mocked '}}]}
        yield {'choices': [{'delta': {'content': 'OpenAI '}}]}
        yield {'choices': [{'delta': {'content': 'response'}}]}

    return generate_chunks()
```

*   `client`: This fixture creates a test client for the Flask app.
    *   `app.config['TESTING'] = True`: Configures the Flask app for testing.
    *   `with app.test_client() as client:`: Creates a test client within a context manager.
    *   `yield client`: Yields the test client to the tests.
*   `mock_openai_response`: This fixture creates a mock OpenAI API response.
    *   It defines a generator function `generate_chunks` that yields a series of mock chunks.
    *   This allows you to simulate a streaming response from the OpenAI API without actually calling the API.

#### Tests

```python
@pytest.mark.asyncio
async def test_chat_endpoint_no_prompt(client):
    response = client.post('/chat', json={}, headers={'Session-Id': 'test_session'})
    assert response.status_code == 400
    data = json.loads(response.data.decode('utf-8'))
    assert data['error'] == "Prompt is required"


@pytest.mark.asyncio
async def test_chat_endpoint_openai(client, mocker, mock_openai_response):
    mocker.patch('backend.app.openai.ChatCompletion.acreate', new_callable=AsyncMock, return_value=mock_openai_response)

    response = client.post(
        '/chat',
        json={'prompt': 'Test prompt', 'llm': 'OpenAI'},
        headers={'Session-Id': 'test_session'}
    )

    assert response.status_code == 200
    # Simulate reading the stream
    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                data = json.loads(decoded_line[5:])
                if 'content' in data:
                    full_response += data['content']

    assert "Mocked OpenAI response" in full_response



@pytest.mark.asyncio
async def test_chat_endpoint_gemini_stub(client):
    response = client.post(
        '/chat',
        json={'prompt': 'Test prompt', 'llm': 'Gemini'},
        headers={'Session-Id': 'test_session'}
    )
    assert response.status_code == 200
    # Simulate reading the stream
    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                data = json.loads(decoded_line[5:])
                if 'content' in data:
                    full_response += data['content']
    assert "Gemini API Stub Response" in full_response


@pytest.mark.asyncio
async def test_clear_context(client):
    # First, create some context
    client.post(
        '/chat',
        json={'prompt': 'Test prompt', 'llm': 'OpenAI'},
        headers={'Session-Id': 'test_session'}
    )
    response = client.post('/clear_context', headers={'Session-Id': 'test_session'})
    assert response.status_code == 200
    data = json.loads(response.data.decode('utf-8'))
    assert data['message'] == "Context cleared"
```

*   `@pytest.mark.asyncio`: This decorator marks the test function as an asynchronous test function.
*   `test_chat_endpoint_no_prompt`: Tests the `/chat` endpoint when no prompt is provided.  It verifies that the endpoint returns a 400 error with the correct error message.  This is a basic but important test to ensure that the input validation is working correctly.
*   `test_chat_endpoint_openai`: Tests the `/chat` endpoint with the OpenAI LLM.
    *   `mocker.patch('backend.app.openai.ChatCompletion.acreate', new_callable=AsyncMock, return_value=mock_openai_response)`: This line uses the `mocker` fixture to mock the `openai.ChatCompletion.acreate` method.  This prevents the test from actually calling the OpenAI API.  Instead, it returns the `mock_openai_response` fixture.
    *   It verifies that the endpoint returns a 200 status code and that the response contains the mocked OpenAI response.
    *   The test simulates reading the stream by iterating over the lines of the response and extracting the content.
*   `test_chat_endpoint_gemini_stub`: Tests the `/chat` endpoint with the Gemini LLM stub.  It verifies that the endpoint returns a 200 status code and that the response contains the stub response.
*   `test_clear_context`: Tests the `/clear_context` endpoint.
    *   First, it creates some context by calling the `/chat` endpoint.
    *   Then, it calls the `/clear_context` endpoint and verifies that it returns a 200 status code and the correct message.

### `backend/Dockerfile`

This file defines the Docker image for the backend.

```docker