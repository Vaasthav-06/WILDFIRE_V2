"""
src/api/routes/chatbot.py

Chatbot endpoint processing user messages with Gemini.
Note: Full RAG architecture will be implemented in Phase 5.
For now, this maintains the original behavior but is extracted into a clean module.
"""

import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
import os

router = APIRouter()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
    )
    chat_session = model.start_chat(history=[])
else:
    logger.warning("GEMINI_API_KEY not found in environment. Chatbot will return dummy responses.")
    model = None


class ChatbotRequest(BaseModel):
    message: str
    fire_context: str = ""


@router.post("/api/chatbot", tags=["Chatbot"])
async def chatbot(request: ChatbotRequest):
    """
    Handle chatbot queries, injecting wildfire context if provided.
    """
    logger.info(f"POST /api/chatbot | message: {request.message[:50]}...")
    
    if model is None:
        return {
            "reply": (
                "Dummy response: I received your message, but the Gemini API key "
                "is not configured in the backend. Context provided: \n"
                f"{request.fire_context}"
            )
        }
    
    # Strict RAG grounding prompt construction
    prompt = f"User Message: {request.message}\n\n"
    if request.fire_context:
        prompt += f"Context regarding the user's selected map cell:\n{request.fire_context}\n\n"
        prompt += "STRICT INSTRUCTION: Do NOT hallucinate fire metrics, probabilities, or weather conditions. Use ONLY the provided context above to answer the user's question. If the user asks about data not present in the context, politely state that you do not have that information."
    else:
        prompt += "STRICT INSTRUCTION: The user has NOT selected a specific location on the map, so you do not have current fire risk data. Politely ask the user to click a cell on the map to provide you with the necessary fire context before you can give specific risk or weather predictions."

        
    try:
        response = chat_session.send_message(prompt)
        return {"reply": response.text}
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {str(e)}")
