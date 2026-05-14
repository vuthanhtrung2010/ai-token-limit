import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from litellm import completion, token_counter
import uvicorn
import dotenv

app = FastAPI()

# --- CONFIGURATION ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_NAME = "qwen/qwen3-coder-480b-a35b-instruct"
VOAI_LIMIT = 4096

# --- OPEN WEBUI DISCOVERY ---
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen3-coder-voai",
            "object": "model",
            "created": 1715641200,
            "owned_by": "nvidia"
        }]
    }

# --- PROXY LOGIC ---
@app.post("/v1/chat/completions")
async def voai_proxy(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    
    # 1. Handle "/check" Command
    if messages and messages[-1]["content"].strip() == "/check":
        tokens_used = token_counter(model="gpt-4", messages=messages[:-1]) # Approximate for Qwen3
        remaining = VOAI_LIMIT - tokens_used
        return {
            "choices": [{
                "message": {"role": "assistant", "content": f"📊 **VOAI Stats:**\n- Tokens Used: `{tokens_used}`\n- Remaining Budget: `{max(0, remaining)}`"},
                "finish_reason": "stop"
            }]
        }

    # 2. Set strict 4k output limit
    data["max_tokens"] = VOAI_LIMIT
    stream = data.get("stream", False)

    if stream:
        async def stream_generator():
            full_content = ""
            for chunk in completion(
                model=MODEL_NAME,
                messages=messages,
                api_base=NVIDIA_BASE_URL,
                api_key=NVIDIA_API_KEY,
                stream=True,
                **{k: v for k, v in data.items() if k not in ["model", "messages", "max_tokens", "stream"]}
            ):
                content = chunk.choices[0].delta.content or ""
                full_content += content
                
                # If model hits the 4k wall
                if chunk.choices[0].finish_reason == "length":
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': '... [TRUNCATED]'}, 'finish_reason': 'stop'}]})}\n\n"
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': '⚠️ **VOAI ERROR:** 4k Output Exceeded. Response discarded.'}, 'finish_reason': 'stop'}]})}\n\n"
                    break
                
                yield f"data: {json.dumps(chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 3. Non-streaming logic
    response = completion(
        model=MODEL_NAME,
        messages=messages,
        api_base=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
        **{k: v for k, v in data.items() if k not in ["model", "messages", "max_tokens"]}
    )

    if response.choices[0].finish_reason == "length":
        response.choices[0].message.content = "⚠️ **VOAI ERROR:** Context exceeded 4k tokens."
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
