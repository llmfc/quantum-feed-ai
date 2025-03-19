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
    assert data['error'] == \"Prompt is required\"


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
    full_response = \"\"
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith(\"data:\"):
                data = json.loads(decoded_line[5:])
                if 'content' in data:
                    full_response += data['content']

    assert \"Mocked OpenAI response\" in full_response



@pytest.mark.asyncio
async def test_chat_endpoint_gemini_stub(client):
    response = client.post(
        '/chat',
        json={'prompt': 'Test prompt', 'llm': 'Gemini', 'model': 'gemini-pro'},
        headers={'Session-Id': 'test_session'}
    )
    assert response.status_code == 200
    # Simulate reading the stream
    full_response = \"\"
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith(\"data:\"):
                data = json.loads(decoded_line[5:])
                if 'content' in data:
                    full_response += data['content']
    assert \"Gemini API Stub Response\" in full_response


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
    assert data['message'] == \"Context cleared\"