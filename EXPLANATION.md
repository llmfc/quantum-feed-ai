**Here you can check all the code explanation.**

Okay, I understand. I will provide a comprehensive explanation of the complete code, including each file and every block in detail, leaving nothing out. I will adopt the persona of a principal software engineer explaining the code to everyone, ensuring clarity and thoroughness.

        ## LLM Chat Application

        This is a simple chat application that leverages Large Language Models (LLMs) to generate responses based on user-provided prompts. It consists of two primary components: a frontend built with React and a backend built with Flask. The application facilitates real-time conversations, allowing users to interact with different LLMs through a user-friendly interface.

        ### Project Structure

        The project is organized into the following directory structure:

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

        ### Backend (`backend/`)

        The backend is responsible for:

        *   Handling API requests from the frontend.
        *   Interacting with different LLMs (currently OpenAI and a Gemini stub).
        *   Managing conversation history for each session.
        *   Providing the core logic for generating and streaming responses.

        #### `backend/app.py`

        This is the primary application file for the backend. It defines the Flask application, API endpoints, and the logic for interacting with the LLMs.

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

            async def generate_response(self, prompt, context, temperature, model):
                raise NotImplementedError

        class OpenAIProvider(LLMProvider):
            def __init__(self):
                super().__init__("OpenAI")

            async def generate_response(self, prompt, context, temperature, model):
                try:
                    messages = [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }]
                    # Include context in the messages
                    for user_prompt, assistant_response in context:
                        messages.append({"role": "user", "content": user_prompt})
                        messages.append({"role": "assistant", "content": assistant_response})
                    messages.append({"role": "user", "content": prompt})

                    logging.info(f"Calling OpenAI API with prompt: {prompt}, model: {model}")

                    response = await openai.ChatCompletion.acreate(
                        model=model,  # Use selected model
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

            async def generate_response(self, prompt, context, temperature, model):
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
            model = data.get("model", "gpt-3.5-turbo") # Default model
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
                        response = await llm.generate_response(prompt, context, temperature, model)
                        async for chunk in response:
                            if chunk and 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:\
                              chunk_message = chunk['choices'][0]['delta']['content']
                              full_response += chunk_message  # Accumulate the response
                              yield f"data: {json.dumps({'content': chunk_message})}\n\n"
                            else:\
                              #Handle edge case where the content is empty\
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

        ##### Imports

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

        *   `os`: Provides functions for interacting with the operating system, such as accessing environment variables.
        *   `json`: Enables encoding and decoding JSON (JavaScript Object Notation) data, which is commonly used for data transmission over the web.
        *   `logging`: Facilitates logging events and errors within the application, crucial for debugging and monitoring. The `logging` module is highly configurable, allowing different levels of logging (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) and various output formats.
        *   `time`: Allows measuring execution time, useful for performance analysis and optimization.
        *   `flask`: The core library for creating the web application. It provides tools and classes for building web applications, including routing, request handling, and response generation.
        *   `request`: Flask's request object, used to access incoming request data (e.g., headers, JSON payload).
        *   `jsonify`: Flask's helper function for creating JSON responses, simplifying the process of converting Python dictionaries into JSON format.
        *   `Response`: Flask's class to handle complex responses, in this case, stream responses. Essential for implementing Server-Sent Events (SSE).
        *   `flask_cors`: Handles Cross-Origin Resource Sharing (CORS), which is essential for allowing requests from the frontend (running on a different domain/port) to access the backend. CORS is a security feature implemented by web browsers to prevent malicious websites from making unauthorized requests to other domains.
        *   `openai`: The OpenAI Python library, providing an interface for interacting with the OpenAI API.
        *   `asyncio`: Enables asynchronous programming, which is vital for handling I/O-bound operations (like API calls) efficiently without blocking the main thread. It allows the server to handle more requests concurrently, improving performance and responsiveness.

        ##### Logging Configuration

        ```python
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        ```

        *   This configures the logging system to output messages at the `INFO` level or higher (e.g., `WARNING`, `ERROR`, `CRITICAL`). The format string specifies the structure of the log messages, including the timestamp (`%(asctime)s`), log level (`%(levelname)s`), and the message itself (`%(message)s`). This configuration provides a standardized and informative logging output.

        ##### Flask App Initialization

        ```python
        app = Flask(__name__)
        CORS(app)
        ```

        *   `app = Flask(__name__)`: Creates a Flask application instance. `__name__` is a special Python variable that represents the name of the current module. Flask uses this to determine the root path of the application.
        *   `CORS(app)`: Enables CORS for the entire application, allowing requests from any origin. This is convenient for development but should be restricted in production to specific origins for security reasons.

        ##### API Key Loading

        ```python
        # Load API keys from environment variables
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Placeholder, not used until Gemini integration

        if not OPENAI_API_KEY:
            logging.error("OpenAI API key not found in environment variables.")
            raise ValueError("OpenAI API key not found in environment variables.")

        openai.api_key = OPENAI_API_KEY
        ```

        *   This section retrieves the OpenAI API key from the environment variable `OPENAI_API_KEY` using `os.environ.get()`. It also includes a placeholder for the Gemini API key, which is currently not used.
        *   It performs a check to ensure that the `OPENAI_API_KEY` is set. If not, it logs an error message using the `logging.error()` function and raises a `ValueError` exception, preventing the application from starting without a valid API key. This is a critical step to ensure proper authentication with the OpenAI API.
        *   `openai.api_key = OPENAI_API_KEY`: Configures the OpenAI library to use the retrieved API key, enabling the application to make authenticated requests to the OpenAI API.

        ##### Conversation History

        ```python
        # In-memory context storage (for demonstration purposes)
        conversation_history = {}  # Session ID -> list of (prompt, response) tuples
        ```

        *   This initializes an in-memory dictionary called `conversation_history` to store the conversation history for each session. The keys of the dictionary are session IDs, and the values are lists of (prompt, response) tuples. This is a simple way to maintain context within a conversation.
        *   **Caveat:** This approach is suitable for demonstration purposes only. Storing conversation history in memory is not scalable or reliable for production environments. If the server restarts, the in-memory history will be lost. A persistent storage mechanism, such as a database (e.g., PostgreSQL, MongoDB), should be used in production to ensure data durability.

        ##### LLM Abstraction Layer

        ```python
        # --- LLM Abstraction Layer ---
        class LLMProvider:
            def __init__(self, name):
                self.name = name

            async def generate_response(self, prompt, context, temperature, model):
                raise NotImplementedError

        class OpenAIProvider(LLMProvider):
            def __init__(self):
                super().__init__("OpenAI")

            async def generate_response(self, prompt, context, temperature, model):
                try:
                    messages = [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }]
                    # Include context in the messages
                    for user_prompt, assistant_response in context:
                        messages.append({"role": "user", "content": user_prompt})
                        messages.append({"role": "assistant", "content": assistant_response})
                    messages.append({"role": "user", "content": prompt})

                    logging.info(f"Calling OpenAI API with prompt: {prompt}, model: {model}")

                    response = await openai.ChatCompletion.acreate(
                        model=model,  # Use selected model
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

            async def generate_response(self, prompt, context, temperature, model):
                # Placeholder response
                await asyncio.sleep(1)  # Simulate some processing time
                return "Gemini API Stub Response"


        llm_providers = {
            "OpenAI": OpenAIProvider(),
            "Gemini": GeminiProvider(),
        }
        ```

        *   This section defines an abstraction layer for interacting with different LLMs, promoting modularity and flexibility.

        *   `LLMProvider`: An abstract base class for LLM providers. It defines the `generate_response` method, which is responsible for generating a response to a prompt. This method raises a `NotImplementedError`, forcing subclasses to implement it. The constructor takes a `name` argument, which stores the name of the LLM provider.

        *   `OpenAIProvider`: A subclass of `LLMProvider` that implements the `generate_response` method for the OpenAI API.
            *   It constructs a list of messages, including a system message ("You are a helpful assistant.") and the conversation history. The conversation history is formatted as a series of user and assistant messages, maintaining the context of the conversation.
            *   It calls the `openai.ChatCompletion.acreate` method to generate a response. The `model`, `messages`, `temperature`, and `stream` parameters are passed to the `acreate` method. The `stream=True` argument enables streaming responses from the OpenAI API, allowing the client to receive the response in chunks.
            *   It includes comprehensive error handling for potential OpenAI API errors, such as rate limits, authentication errors, API connection errors, and invalid requests. Each error is logged with detailed information, and a user-friendly exception is raised to provide informative error messages to the client.

        *   `GeminiProvider`: A placeholder implementation for the Gemini API. Currently, it simply returns a stub response after a short delay. This class serves as a template for future integration with the Gemini API.

        *   `llm_providers`: A dictionary that maps LLM names (e.g., "OpenAI", "Gemini") to their corresponding provider instances. This allows the application to easily switch between different LLMs by selecting the appropriate provider from the dictionary.

        ##### `/chat` Endpoint

        ```python
        # --- API Endpoints ---
        @app.route("/chat", methods=["POST"])
        async def chat():
            data = request.get_json()
            prompt = data.get("prompt")
            llm_name = data.get("llm", "OpenAI")  # Default to OpenAI
            temperature = data.get("temperature", 0.7)  # Default temperature
            model = data.get("model", "gpt-3.5-turbo") # Default model
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
                        response = await llm.generate_response(prompt, context, temperature, model)
                        async for chunk in response:
                            if chunk and 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:\
                              chunk_message = chunk['choices'][0]['delta']['content']
                              full_response += chunk_message  # Accumulate the response
                              yield f"data: {json.dumps({'content': chunk_message})}\n\n"
                            else:\
                              #Handle edge case where the content is empty\
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

        *   This endpoint handles incoming chat requests from the frontend.

        *   `@app.route("/chat", methods=["POST"])`: Defines the route for the chat endpoint. It specifies that the endpoint only accepts POST requests.

        *   `data = request.get_json()`: Parses the JSON payload from the request body, extracting the data sent by the client.

        *   `prompt = data.get("prompt")`: Retrieves the `prompt` from the parsed JSON data. The `prompt` contains the user's input message.

        *   `llm_name = data.get("llm", "OpenAI")`: Retrieves the selected LLM name from the JSON data, defaulting to "OpenAI" if no LLM name is provided. This allows the user to choose which LLM to use for generating the response.

        *   `temperature = data.get("temperature", 0.7)`: Retrieves the `temperature` from the JSON data, defaulting to 0.7 if no temperature is provided. The `temperature` parameter controls the randomness of the generated responses. A higher temperature will result in more random responses, while a lower temperature will result in more deterministic responses.

        *   `model = data.get("model", "gpt-3.5-turbo")`: Retrieves the selected model from the JSON data, defaulting to "gpt-3.5-turbo" if no model is provided. This allows the user to specify which OpenAI model to use.

        *   `session_id = request.headers.get("Session-Id")`: Retrieves the session ID from the `Session-Id` header. The session ID is used to maintain the conversation history for each user. Using headers for passing the session ID is a common practice, although cookies are another option.

        *   The code then performs input validation to ensure that the prompt, LLM name, and session ID are provided. If any of these are missing or invalid, it returns a 400 error with a JSON payload indicating the error. Input validation is crucial for preventing errors and security vulnerabilities.

        *   `if session_id not in conversation_history: conversation_history[session_id] = []`: Retrieves or creates the conversation history for the session. If a session ID is not found in the `conversation_history` dictionary, a new entry is created with an empty list as the value.

        *   `context = conversation_history[session_id]`: Gets the conversation history for the current session. The conversation history is used as context for the LLM, allowing it to generate more relevant and coherent responses.

        *   The code then calls the selected LLM to generate a response:

            *   `llm = llm_providers[llm_name]`: Retrieves the LLM provider instance based on the `llm_name`.
            *   `response = await llm.generate_response(prompt, context, temperature, model)`: Calls the `generate_response` method of the selected LLM provider, passing the prompt, context, temperature, and model as arguments.

        *   The code then streams the response back to the client using Server-Sent Events (SSE):

            *   The code defines an inner asynchronous generator function `generate()`:
                *   `nonlocal full_response`: Allows the inner function to modify the `full_response` variable in the outer scope.
                *   `response = await llm.generate_response(prompt, context, temperature, model)`: Calls the `generate_response` method of the selected LLM provider.
                *   `async for chunk in response:`: Iterates over the chunks of the streamed response.

                *   The code checks for `chunk`, `'choices' in chunk`, `len(chunk['choices']) > 0`, `'delta' in chunk['choices'][0]` and `'content' in chunk['choices'][0]['delta']` to ensure the structure is as expected before extracting data. This is a form of defensive programming, guarding against unexpected data structures from the OpenAI API.

                *   `chunk_message = chunk['choices'][0]['delta']['content']`: Extracts the content of the chunk.

                *   `full_response += chunk_message`: Appends the chunk to the `full_response` variable, accumulating the complete response from the LLM.
                *   `yield f"data: {json.dumps({'content': chunk_message})}\n\n"`: Yields the chunk as an SSE event. SSE is a simple protocol for pushing real-time updates from the server to the client. The data is formatted as `data: {JSON payload}\n\n`, which is the standard SSE format.

                *   The `finally` block ensures that a `[DONE]` event is sent to the client when the stream is complete, regardless of whether an error occurred. This signals to the client that the response is finished.
                *   Error handling included to catch generic exceptions and stream error messages to the client to notify about backend errors.

            *   `response = Response(generate(), mimetype='text/event-stream')`: Creates a Flask `Response` object with the `generate` function as the generator and sets the `mimetype` to `text/event-stream`. This tells the client that the response will be streamed using SSE.

        *   The `try...except...finally` block ensures that the conversation history is updated even if an error occurs during the response generation. The `finally` block is always executed, regardless of whether an exception is raised.

        *   `conversation_history[session_id].append((prompt, full_response))`: Updates the conversation history with the new prompt and response. This allows the LLM to maintain context in subsequent turns of the conversation.

        *   `conversation_history[session_id] = conversation_history[session_id][-5:]`: Limits the conversation history to the last 5 turns. This helps to prevent the context from growing too large and exceeding the LLM's context window, which can lead to performance issues and less relevant responses. This is a form of memory management for the conversation.

        *   The endpoint also includes timing information, logging the time it takes to generate the full response. This is useful for monitoring the performance of the LLM and identifying potential bottlenecks.

        ##### `/clear_context` Endpoint

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

        *   This endpoint clears the conversation history for a given session.

        *   `@app.route("/clear_context", methods=["POST"])`: Defines the route for the clear context endpoint. It only accepts POST requests.

        *   `session_id = request.headers.get("Session-Id")`: Retrieves the session ID from the `Session-Id` header, identifying the session to clear.

        *   `if session_id in conversation_history:`: Checks if the session ID exists in the conversation history.

        *   `del conversation_history[session_id]`: Deletes the conversation history for the session, effectively resetting the conversation.

        *   It returns a JSON response indicating whether the context was cleared.

        ##### Main Block

        ```python
        if __name__ == "__main__":
            app.run(debug=True, port=5000)
        ```

        *   This block starts the Flask development server when the script is executed directly. The `if __name__ == "__main__":` condition ensures that the code within the block is only executed when the script is run as the main program, not when it is imported as a module.

        *   `app.run(debug=True, port=5000)`: Starts the Flask development server. The `debug=True` argument enables the Flask debugger, which provides helpful error messages and allows the server to automatically reload when changes are made to the code. **Important:** Debug mode should be disabled in production environments for security reasons. The `port=5000` argument specifies the port on which the server will listen for incoming requests.

        #### `backend/tests/test_app.py`

        This file contains unit tests for the backend application, ensuring the correctness and reliability of the API endpoints.

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
                json={'prompt': 'Test prompt', 'llm': 'OpenAI', 'model': 'gpt-3.5-turbo'},
                headers={'Session-Id': 'test_session'}
            )

            assert response.status_code == 200
            # Simulate reading the stream
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):\
                        data = json.loads(decoded_line[5:])
                        if 'content' in data:
                            full_response += data['content']

            assert "Mocked OpenAI response" in full_response



        @pytest.mark.asyncio
        async def test_chat_endpoint_gemini_stub(client):
            response = client.post(
                '/chat',
                json={'prompt': 'Test prompt', 'llm': 'Gemini', 'model': 'gemini-pro'},
                headers={'Session-Id': 'test_session'}
            )
            assert response.status_code == 200
            # Simulate reading the stream
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):\
                        data = json.loads(decoded_line[5:])
                        if 'content' in data:
                            full_response += data['content']
            assert "Gemini API Stub Response" in full_response


        @pytest.mark.asyncio
        async def test_clear_context(client):
            # First, create some context
            client.post(
                '/chat',
                json={'prompt': 'Test prompt', 'llm': 'OpenAI', 'model': 'gpt-3.5-turbo'},
                headers={'Session-Id': 'test_session'}
            )
            response = client.post('/clear_context', headers={'Session-Id': 'test_session'})
            assert response.status_code == 200
            data = json.loads(response.data.decode('utf-8'))
            assert data['message'] == "Context cleared"
        ```

        ##### Imports

        ```python
        import pytest
        import json
        from unittest.mock import AsyncMock, MagicMock
        from backend.app import app
        ```

        *   `pytest`: The pytest testing framework, which provides a powerful and flexible environment for writing and running tests. pytest simplifies the process of test discovery, execution, and reporting.
        *   `json`: For working with JSON data, used for serializing and deserializing data in the tests.
        *   `unittest.mock`: Provides tools for mocking objects and functions during testing, allowing you to isolate the code being tested from external dependencies. Mocking is essential for writing unit tests that are fast, reliable, and independent of external systems.
        *   `AsyncMock`: Used for mocking asynchronous functions, ensuring that the tests can properly simulate asynchronous behavior.
        *   `MagicMock`: A versatile class for general mocking purposes, allowing you to create mock objects with customizable behavior.
        *   `backend.app`: Imports the Flask app instance from `backend/app.py`, allowing the tests to interact with the application's API endpoints.

        ##### Fixtures

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
                yield {'choices': [{'delta': {'content': 'OpenAI '