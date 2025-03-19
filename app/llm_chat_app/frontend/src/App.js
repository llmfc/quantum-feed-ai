import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
    const [prompt, setPrompt] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [llm, setLlm] = useState('OpenAI');
    const [model, setModel] = useState('gpt-3.5-turbo'); // Default model
    const [temperature, setTemperature] = useState(0.7);
    const [sessionId, setSessionId] = useState(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [currentStreamedResponse, setCurrentStreamedResponse] = useState('');
    const chatHistoryRef = useRef(null);
    const [isLoading, setIsLoading] = useState(false);

    const llmModels = {
        'OpenAI': [
            'gpt-3.5-turbo',
            'gpt-4',
            'gpt-4-turbo-preview'
        ],
        'Gemini': [
            'gemini-pro',
            'gemini-1.5-pro-latest'
        ],
        'Mistral': [
            'mistral-tiny',
            'mistral-small',
            'mistral-medium'
        ],
        'O1': [
            'oasst-sft-1-pythia-12b',
            'oasst-sft-6-llama-30b'
        ],
        'LLama': [
            'Llama-2-7b-chat-hf',
            'Llama-2-13b-chat-hf'
        ]
    };

    useEffect(() => {
        const storedChatHistory = localStorage.getItem('chatHistory');
        if (storedChatHistory) {
            setChatHistory(JSON.parse(storedChatHistory));
        }

        let session = localStorage.getItem('sessionId');
        if (!session) {
            session = generateSessionId();
            localStorage.setItem('sessionId', session);
        }
        setSessionId(session);
    }, []);

    useEffect(() => {
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    }, [chatHistory]);

    const generateSessionId = () => {
        return Math.random().toString(36).substring(2, 15);
    };

    const handleLlmChange = (event) => {
        const selectedLlm = event.target.value;
        setLlm(selectedLlm);
        // Reset model to the first option when LLM changes
        setModel(llmModels[selectedLlm][0]);
    };

    const handleModelChange = (event) => {
        setModel(event.target.value);
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
            return;
        }

        setIsStreaming(true);
        setCurrentStreamedResponse('');
        setIsLoading(true);

        const newChatHistory = [...chatHistory, { type: 'user', text: prompt }];
        setChatHistory(newChatHistory);

        try {
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                    'Session-Id': sessionId,
                },
                body: JSON.stringify({ prompt: prompt, llm: llm, temperature: temperature, model: model }),
                credentials: 'omit',
            });

            if (!response.ok) {
                const errorData = await response.json();
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
                                console.error(\"Error from backend:\", parsedData.error);
                                setChatHistory(prev => [...prev, { type: 'error', text: `Error: ${parsedData.error}` }]);
                                setIsStreaming(false);
                                reader.cancel();
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
            setIsLoading(false);
        }

        setPrompt('');
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
                    'Session-Id': sessionId,
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
            <h1 className=\"app-title\">LLM Chat Application</h1>
            <div className=\"model-options\">
                <div className=\"llm-selector\">
                    <label htmlFor=\"llm\">LLM:</label>
                    <select id=\"llm\" value={llm} onChange={handleLlmChange}>
                        {Object.keys(llmModels).map((llmName) => (
                            <option key={llmName} value={llmName}>{llmName}</option>
                        ))}
                    </select>
                </div>

                <div className=\"model-selector\">
                    <label htmlFor=\"model\">Model:</label>
                    <select id=\"model\" value={model} onChange={handleModelChange}>
                        {llmModels[llm].map((modelName) => (
                            <option key={modelName} value={modelName}>{modelName}</option>
                        ))}
                    </select>
                </div>
            </div>

            <div className=\"chat-container\">
                <div className=\"chat-history\" ref={chatHistoryRef}>
                    {chatHistory.map((message, index) => (
                        <div key={index} className={`message ${message.type}`}>
                            <div className=\"message-content\">
                                {message.text}
                            </div>
                        </div>
                    ))}
                    {isStreaming && (
                        <div className=\"message bot\">
                            <div className=\"message-content\">
                                {currentStreamedResponse}
                            </div>
                        </div>
                    )}
                    {isLoading && <div className=\"loading-spinner\">Loading...</div>}
                </div>
                <form onSubmit={handleSubmit} className=\"prompt-input\">
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
                <div className=\"controls\">
                    <div className=\"temperature-control\">
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
                    <button onClick={handleClearContext} className=\"new-chat-button\">New Chat</button>
                </div>
            </div>
        </div>
    );
}

export default App;