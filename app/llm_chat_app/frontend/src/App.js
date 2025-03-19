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
            <div className=\"error-boundary\">
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
                while ((eventEndIndex = buffer.indexOf('\
\
')) !== -1) {
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
                                console.error(\"Error from backend:\", parsedData.error);
                                setChatHistory(prev => [...prev, { type: 'error', text: `Error: ${parsedData.error}` }]);
                                setIsStreaming(false);
                                reader.cancel(); // Stop reading stream on error
                                break;
                            }
                        } catch (parseError) {
                            console.error(\"Error parsing SSE data:\", parseError, \"Data:\", data);
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
        <div className=\"App\">
            <h1>LLM Chat Application</h1>
            <div className=\"controls\">
                <div>
                    <label htmlFor=\"llm\">LLM:</label>
                    <select id=\"llm\" value={llm} onChange={handleLlmChange}>
                        <option value=\"OpenAI\">OpenAI</option>
                        <option value=\"Gemini\">Gemini</option>
                    </select>
                </div>
                <div>
                    <label htmlFor=\"temperature\">Temperature:</label>
                    <input
                        type=\"range\"
                        id=\"temperature\"
                        min=\"0\"
                        max=\"1\"
                        step=\"0.01\"
                        value={temperature}
                        onChange={handleTemperatureChange}
                    />
                    <span>{temperature}</span>
                </div>
            </div>
            <div className=\"chat-history\" ref={chatHistoryRef}>
                {chatHistory.map((message, index) => (
                    <div key={index} className={`message ${message.type}`}>
                        {message.text}
                    </div>
                ))}
                {isStreaming && (
                    <div className=\"message bot\">
                        {currentStreamedResponse}
                    </div>
                )}
                {isLoading && <div className=\"loading-spinner\">Loading...</div>} {/* Show loading spinner */}
            </div>
            <form onSubmit={handleSubmit}>
                <input
                    type=\"text\"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder=\"Enter your prompt...\"
                    disabled={isStreaming}
                />
                <button type=\"submit\" disabled={isStreaming || isLoading}>
                    {isStreaming || isLoading ? 'Sending...' : 'Send'}
                </button>
            </form>
            <button onClick={handleClearContext}>New Chat</button>
        </div>
    );
}

export default App;