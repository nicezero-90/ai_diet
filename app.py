import os
from dotenv import load_dotenv
from logger import logger
import json
import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langsmith.run_trees import RunTree
from openai_service import handle_ai_diet_chat
from client_middleware import process_client_request, extract_client_id_from_token, extract_token
from conversation_service import (
    get_or_create_conversation,
    add_messages_to_conversation,
    get_last_response_id
)

load_dotenv()
# LangSmith configuration
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANCHAIN_PROJECT") or "ai-diet-1.0.1-v1"

logger.info("Logger initialized.")

app = FastAPI(title="AI Diet API", description="Dietary analysis and recording API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def remove_system_message_in_messages(messages: List[Dict]) -> List[Dict]:
    """
    Remove system messages from the list of messages.
    
    Args:
        messages: List of message objects in the original format
        
    Returns:
        List of message objects, potentially without system messages
    """
    if not messages or not isinstance(messages, list):
        return messages
    
    filtered_messages = [message for message in messages if message.get("role") != "system"]
    return filtered_messages

async def format_input_image_in_messages(messages: List[Dict]) -> List[Dict]:
    """
    Convert image_url type items in user messages to input_image format.
    
    Args:
        messages: List of message objects in the original format
        
    Returns:
        List of message objects with converted image formats
    """
    if not messages or not isinstance(messages, list):
        return messages
    
    formatted_messages = []
    
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            new_content = []
            
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url" and "image_url" in item:
                    # convert from 
                    # {"type": "image_url", "image_url": {"url": "url"}}
                    # to
                    # {"type": "input_image", "image_url": "url"}
                    if isinstance(item["image_url"], dict) and "url" in item["image_url"]:
                        image_url = item["image_url"]["url"]
                    else:
                        image_url = item["image_url"]
                    
                    # 創建新的input_image格式
                    new_content.append({
                        "type": "input_image",
                        "image_url": image_url
                    })
                elif isinstance(item, dict) and item.get("type") == "text":
                    # convert from 
                    # {"type": "text", "text": "text"}
                    # to
                    # {"type": "input_text", "text": "text"}
                    new_content.append({
                        "type": "input_text",
                        "text": item.get("text", "")
                    })
                else:
                    # 保留其他類型的內容
                    new_content.append(item)
            
            # 更新消息的內容
            formatted_message = message.copy()
            formatted_message["content"] = new_content
            formatted_messages.append(formatted_message)
        else:
            # 保留非用戶消息或不含列表內容的消息
            formatted_messages.append(message)
    
    return formatted_messages

@app.options("/ai-diet")
async def options_ai_diet():
    """Handle CORS preflight requests."""
    return Response(
        content="",
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Token, X-Client-ID",
            "Access-Control-Max-Age": "3600"
        }
    )

@app.post("/ai-diet")
async def ai_diet(request: Request):
    """Handle diet analysis and forwarding requests using Agent & Runner."""
    # Create a root LangSmith run for the request
    root_run = RunTree(
        name="ai_diet",
        run_type="chain",
        project_name="ai-diet-1.0.1-v1",
        inputs={"request": "Incoming request"}
    )
    
    try:
        # Get request data and headers
        request_data = await request.json()
        headers = dict(request.headers)
        conversation_id = request_data.get("conversation_id")
        logger.info(f"Received request with conversation_id: {conversation_id}")
        
        # Extract the latest user message
        messages = request_data.get("messages", [])
        
        # Extract client ID for conversation tracking
        token = await extract_token(headers)
        client_id = await extract_client_id_from_token(token) if token else None
        
        # Get or create conversation
        conversation = await get_or_create_conversation(conversation_id, client_id)
        logger.info(f"Conversation retrieved or created: {conversation}")
        conversation_id = str(conversation["_id"])
        is_first_request = len(conversation.get("messages", [])) == 0

        # Format and filter messages
        formatted_messages = await format_input_image_in_messages(messages)
        if is_first_request:
            formatted_messages = await remove_system_message_in_messages(formatted_messages)
            formatted_messages, client_info, client_targets = await process_client_request(
                headers=headers, 
                messages=formatted_messages, 
                parent_run=root_run
            )
        # logger.info(f"Client information processed, client_id: {client_info.get('client_id', 'unknown')}")

        # Add the current message to the conversation database
        await add_messages_to_conversation(
            conversation_id=conversation_id,
            messages=formatted_messages
        )
        
        # Get the last response ID for continuity
        previous_response_id = await get_last_response_id(conversation_id)
        
        # Handle the AI diet chat with the enriched data
        response_generator = handle_ai_diet_chat(
            messages=formatted_messages, 
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            parent_run=root_run
        )
        
        # Return streaming response
        return StreamingResponse(
            response_generator,
            media_type="text/event-stream",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        logger.error(error_message)
        
        # End run tree with error
        root_run.end(error=error_message, outputs={"status": 500})
        root_run.post()
        
        # Return streaming error response
        async def generate_error():
            yield f"data: {json.dumps({'error': error_message, 'status': 500})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_error(),
            media_type="text/event-stream",
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
