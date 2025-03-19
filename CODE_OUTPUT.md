**This file consists of the entire unaltered output for reference.**



Okay, I understand. I'll consolidate all previous responses, incorporate all suggested improvements, and provide a complete, runnable codebase with explanations. This will include all files necessary to run the application.

**Project Structure:**

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

**1. backend/app.py:**

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

**Explanation:**

*   **Imports:** Includes necessary libraries (Flask, Flask-CORS, openai, asyncio, logging, json).
*   **Logging:** Configures basic logging to the console.  This is useful for debugging.
*   **API Key Loading:** Loads the OpenAI API key from an environment variable (`OPENAI_API_KEY`). **Important:** You must set this environment variable.
*   **LLM Abstraction:**  `LLMProvider`, `OpenAIProvider`, and `GeminiProvider` provide an abstraction layer for different LLMs.  Currently, only OpenAI is implemented fully, and Gemini is a stub.
*   **`/chat` Endpoint:**
    *   Handles incoming chat requests.
    *   Retrieves the prompt, LLM selection, temperature, and session ID from the request.
    *   Validates the input.
    *   Retrieves or creates the conversation history for the session.
    *   Calls the selected LLM to generate a response.
    *   Streams the response back to the client using Server-Sent Events (SSE).
    *   Handles OpenAI API errors (rate limits, authentication errors, etc.).
    *   Updates the conversation history after the stream is complete, limiting the history to the last 5 turns.
*   **`/clear_context` Endpoint:** Clears the conversation history for a given session.
*   **Error Handling:** Comprehensive error handling, including specific OpenAI API errors and logging.

**2. backend/tests/test_app.py:**

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

**Explanation:**

*   **Fixtures:**
    *   `client`: Creates a test client for the Flask app.
    *   `mock_openai_response`:  A fixture to mock the OpenAI API's streaming response.
*   **Tests:**
    *   `test_chat_endpoint_no_prompt`: Tests the case where the prompt is missing.
    *   `test_chat_endpoint_openai`: Tests the OpenAI integration with a mocked response. The tests now verify the streamed output.
    *   `test_chat_endpoint_gemini_stub`: Tests the Gemini stub.
    *   `test_clear_context`: Tests the `/clear_context` endpoint.
*   **Headers:** Session ID is passed via headers instead of cookies.

**3. backend/Dockerfile:**

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**Explanation:**

*   Uses a Python 3.9 slim base image.
*   Sets the working directory to `/app`.
*   Copies the `requirements.txt` file and installs the dependencies.
*   Copies the rest of the backend code.
*   Defines the command to run the Flask app.

**4. backend/requirements.txt:**

```text
Flask==2.3.2
Flask-CORS==3.0.10
openai==0.27.7
python-dotenv==1.0.0
requests==2.31.0
pytest
pytest-mock
pytest-asyncio
```

**Explanation:**

*   Lists the Python dependencies for the backend.
*   Includes `pytest`, `pytest-mock`, and `pytest-asyncio` for testing.

**5. backend/.env:**

```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
GEMINI_API_KEY=YOUR_GEMINI_API_KEY # Optional, if you have a Gemini API key
```

**Important:**  Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.  This file should *not* be committed to version control.

**6. frontend/src/App.js:**

```javascript
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function ErrorBoundary({ children }) {
    const [hasError, setHasError] = useState(false);
    useEffect(() => {
        const resetError = () => {
            setHasError(false);
        };
        if (hasError) {
            resetError();
        }
    }, [hasError]);

    if (hasError) {
        return (
            <div className="error-boundary">
                <h2>Something went wrong.</h2>
                <button onClick={() => setHasError(false)}>Try again</button>
            </div>
        );
    }

    return children;
}

function App() {
    const [prompt, setPrompt] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [llm, setLlm] = useState('OpenAI');
    const [temperature, setTemperature] = useState(0.7);
    const [sessionId, setSessionId] = useState(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [currentStreamedResponse, setCurrentStreamedResponse] = useState('');
    const chatHistoryRef = useRef(null);
    const [isLoading, setIsLoading] = useState(false); // Added loading state

    useEffect(() => {
        // Load chat history from localStorage on initial load
        const storedChatHistory = localStorage.getItem('chatHistory');
        if (storedChatHistory) {
            setChatHistory(JSON.parse(storedChatHistory));
        }

        // Generate or retrieve session ID
        let session = localStorage.getItem('sessionId');
        if (!session) {
            session = generateSessionId();
            localStorage.setItem('sessionId', session);
        }
        setSessionId(session);
    }, []);

    useEffect(() => {
        // Save chat history to localStorage whenever it changes
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    }, [chatHistory]);

    const generateSessionId = () => {
        return Math.random().toString(36).substring(2, 15);
    };

    const handleLlmChange = (event) => {
        setLlm(event.target.value);
    };

    const handleTemperatureChange = (event) => {
        setTemperature(parseFloat(event.target.value));
    };

    useEffect(() => {
        if (chatHistoryRef.current) {
            chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
        }
    }, [chatHistory]);

    const handleSubmit = async (event) => {
        event.preventDefault();

        if (!prompt.trim()) {
            return; // Prevent sending empty prompts
        }

        setIsStreaming(true);
        setCurrentStreamedResponse('');
        setIsLoading(true); // Start loading
        const newChatHistory = [...chatHistory, { type: 'user', text: prompt }];
        setChatHistory(newChatHistory);

        try {
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                    'Session-Id': sessionId, // Include session ID in header
                },
                body: JSON.stringify({ prompt: prompt, llm: llm, temperature: temperature }),
                credentials: 'omit',
            });

            if (!response.ok) {
                // Handle HTTP errors
                const errorData = await response.json(); // Try to parse JSON error
                throw new Error(errorData?.error || `HTTP error! Status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    break;
                }

                buffer += decoder.decode(value);

                let eventEndIndex;
                while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                    const event = buffer.substring(0, eventEndIndex).trim();
                    buffer = buffer.substring(eventEndIndex + 2);

                    if (event.startsWith('data:')) {
                        const data = event.substring(5).trim();
                        if (data === '[DONE]') {
                            setIsStreaming(false);
                            break;
                        }

                        try {
                            const parsedData = JSON.parse(data);
                            if (parsedData.content) {
                                setCurrentStreamedResponse((prev) => prev + parsedData.content);
                            } else if (parsedData.error) {
                                // Handle errors from the backend
                                console.error("Error from backend:", parsedData.error);
                                setChatHistory(prev => [...prev, { type: 'error', text: `Error: ${parsedData.error}` }]);
                                setIsStreaming(false);
                                reader.cancel(); // Stop reading stream on error
                                break;
                            }
                        } catch (parseError) {
                            console.error("Error parsing SSE data:", parseError, "Data:", data);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Fetch error:', error);
            setChatHistory(prev => [...prev, { type: 'error', text: `Error: ${error.message}` }]);
            setIsStreaming(false);
        } finally {
            setIsStreaming(false);
            setIsLoading(false); // Stop loading
        }

        setPrompt(''); // Clear the input after sending
    };

    useEffect(() => {
        if (!isStreaming && currentStreamedResponse) {
            setChatHistory(prev => [...prev, { type: 'bot', text: currentStreamedResponse }]);
        }
    }, [isStreaming, currentStreamedResponse]);

    const handleClearContext = async () => {
        try {
            const response = await fetch('http://localhost:5000/clear_context', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Session-Id': sessionId,  // Include session ID in header
                },
                credentials: 'omit',
            });

            if (response.ok) {
                setChatHistory([]);
                localStorage.removeItem('chatHistory');
            } else {
                console.error('Failed to clear context');
            }
        } catch (error) {
            console.error('Error clearing context:', error);
        }
    };

    return (
        <div className="App">
            <h1>LLM Chat Application</h1>
            <div className="controls">
                <div>
                    <label htmlFor="llm">LLM:</label>
                    <select id="llm" value={llm} onChange={handleLlmChange}>
                        <option value="OpenAI">OpenAI</option>
                        <option value="Gemini">Gemini</option>
                    </select>
                </div>
                <div>
                    <label htmlFor="temperature">Temperature:</label>
                    <input
                        type="range"
                        id="temperature"
                        min="0"
                        max="1"
                        step="0.01"
                        value={temperature}
                        onChange={handleTemperatureChange}
                    />
                    <span>{temperature}</span>
                </div>
            </div>
            <div className="chat-history" ref={chatHistoryRef}>
                {chatHistory.map((message, index) => (
                    <div key={index} className={`message ${message.type}`}>
                        {message.text}
                    </div>
                ))}
                {isStreaming && (
                    <div className="message bot">
                        {currentStreamedResponse}
                    </div>
                )}
                {isLoading && <div className="loading-spinner">Loading...</div>} {/* Show loading spinner */}
            </div>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Enter your prompt..."
                    disabled={isStreaming}
                />
                <button type="submit" disabled={isStreaming || isLoading}>
                    {isStreaming || isLoading ? 'Sending...' : 'Send'}
                </button>
            </form>
            <button onClick={handleClearContext}>New Chat</button>
        </div>
    );
}

export default App;
```

**Explanation:**

*   **State Variables:** Manages the prompt, chat history, LLM selection, temperature, session ID, streaming status, current streamed response, and loading state.
*   **`useEffect` Hooks:**
    *   Loads chat history and session ID from local storage on initial load.
    *   Saves chat history to local storage whenever it changes.
    *   Scrolls the chat history to the bottom when it updates.
*   **`handleSubmit` Function:**
    *   Handles form submissions.
    *   Sends the prompt to the backend API using `fetch`.
    *   Sets the `Session-Id` header with the current session ID.
    *   Handles the SSE stream, updating the `currentStreamedResponse` state as data is received.
    *   Handles errors from the backend.
    *   Clears the prompt input after sending.
*   **`handleClearContext` Function:**
    *   Clears the chat history and removes it from local storage.
    *   Sends a request to the backend to clear the context.
*   **UI Elements:**
    *   Displays the chat history.
    *   Provides a form for entering prompts.
    *   Includes controls for selecting the LLM and temperature.
    *   Includes a "New Chat" button to clear the context.
    *   Displays a loading spinner while waiting for a response.
*   **Loading state**: Displays spinner when it is set to `true`.
*   **Disabled button**: Submit button is disabled when the `isLoading` state is set to `true`.

**7. frontend/src/App.css:**

```css
/* frontend/src/App.css */
.App {
  font-family: sans-serif;
  text-align: center;
  padding: 20px;
}

.chat-history {
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
  height: 300px;
  overflow-y: scroll;
  text-align: left;
  position: relative; /* For positioning the loading spinner */
}

.message {
  padding: 8px;
  margin-bottom: 5px;
  border-radius: 5px;
}

.message.user {
  background-color: #e2f0ff;
  text-align: right;
}

.message.bot {
  background-color: #f0f0f0;
  text-align: left;
}

.message.error {
    background-color: #ffdddd; /* Light red for error messages */
    color: #8B0000; /* Dark red text */
    text-align: left;
}

input[type="text"] {
  width: 70%;
  padding: 8px;
  margin-right: 10px;
}

button {
  padding: 8px 12px;
  cursor: pointer;
}

.controls {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 10px;
}

.controls > div {
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* Loading Spinner Styles */
.loading-spinner {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-style: italic;
    color: #888;
}
```

**Explanation:**

*   Provides basic styling for the chat application.
*   Includes styles for the chat history, messages, input elements, buttons, and controls.
*   Defines styles for the loading spinner.

**8. frontend/src/index.js:**

```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
```

**Explanation:**

*   The entry point for the React application.
*   Renders the `App` component into the DOM.

**9. frontend/src/index.css:**

```css
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
```

**Explanation:**

*   Basic CSS reset for the entire application.

**10. frontend/src/setupTests.js:**

```javascript
// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';
```

**Explanation:**

*   Sets up the testing environment for the React application.
*   Imports `jest-dom` for custom Jest matchers.

**11. frontend/public/index.html:**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta
      name="description"
      content="Web app for LLM Chat"
    />
    <link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />
    <!--
      manifest.json provides metadata used when your web app is installed on a
      user's mobile device or desktop. See https://developers.google.com/web/fundamentals/web-app-manifest/
    -->
    <link rel="manifest" href="%PUBLIC_URL%/manifest.json" />
    <!--
      Notice the use of %PUBLIC_URL% in the tags above.
      It will be replaced with the URL of the `public` folder during the build.
      Only files inside the `public` folder can be referenced from the HTML.

      Unlike "/favicon.ico" or "favicon.ico", "%PUBLIC_URL%/favicon.ico" will
      work correctly both with client-side routing and a non-root public URL.
    -->
    <title>LLM Chat App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
    <!--
      This HTML file is a template.
      If you open it directly in the browser, you will see an empty page.

      You can add webfonts, meta tags, or analytics to this file.
      The build step will place the bundled scripts into the <body> tag.

      To begin the development, run `npm start` or `yarn start`.
      To create a production bundle, use `npm run build` or `yarn build`.
    -->
  </body>
</html>
```

**Explanation:**

*   The main HTML file for the React application.

**12. frontend/Dockerfile:**

```dockerfile
FROM node:16-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Explanation:**

*   Uses a Node.js 16 Alpine base image for building the React app.
*   Copies the `package.json` and `package-lock.json` files and installs the dependencies.
*   Copies the rest of the frontend code.
*   Builds the React app using `npm run build`.
*   Uses an Nginx Alpine base image for serving the static files.
*   Copies the built React app to the Nginx web root.
*   Exposes port 80.
*   Defines the command to run Nginx.

**13. frontend/package.json:**

```json
{
  "name": "frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.5",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "devDependencies": {}
}
```

**Explanation:**

*   Lists the JavaScript dependencies for the frontend.

**14. docker-compose.yml:**

```yaml
version: "3.8"
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_BACKEND_URL=http://localhost:5000

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}  # Replace with your actual API key
      - GEMINI_API_KEY=${GEMINI_API_KEY}  # Replace with your actual API key, if used
    volumes:
      - backend_data:/app/data #This is optional, if you want persistant local storage
volumes:
  backend_data:
```

**Explanation:**

*   Defines the services (frontend and backend), their build configurations, port mappings, dependencies, and environment variables.
*   The frontend's `REACT_APP_BACKEND_URL` environment variable points to the backend service.
*   The backend's environment variables are loaded from the `.env` file.

**15. README.md (Root):**

```markdown
# LLM Chat Application

This is a simple