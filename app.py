from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoModelForCausalLM, AutoTokenizer

import logging

app = FastAPI()

# Add logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

handler = logging.StreamHandler()

handler.setFormatter(formatter)

logger.addHandler(handler)

# Add CORS

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

intent_model = AutoModelForCausalLM.from_pretrained("llmware/slim-intent")
intent_tokenizer = AutoTokenizer.from_pretrained("llmware/slim-intent")

sentiment_model = AutoModelForCausalLM.from_pretrained("llmware/slim-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("llmware/slim-sentiment")

def getResponse(model, tokenizer, text, params):
  function = "classify"

  prompt = "<human>: " + text + "\n" + f"<{function}> {params} </{function}>\n<bot>:"

  inputs = tokenizer(prompt, return_tensors="pt")
  start_of_input = len(inputs.input_ids[0])

  outputs = model.generate(
      inputs.input_ids.to('cpu'),
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.eos_token_id,
      do_sample=True,
      temperature=0.3,
      max_new_tokens=100
  )

  output = tokenizer.decode(outputs[0][start_of_input:], skip_special_tokens=True)

  return output

@app.get("/")
def read_root():
    return {
        "message": "API running successfully",
        "endpoints": [
            "/api/sentiment/",
            "/api/intent/"
        ]
    }

@app.post("/api/intent/")
def intentResponse(text: str):
  params = "intent"
  try:
    responses = getResponse(intent_model, intent_tokenizer, text, params)
    return responses
  except Exception as e:
    logger.exception(e)
    return {"API Error": str(e)}

@app.post("/api/sentiment/")
def sentimentResponse(text: str):
  params = "sentiment"
  try:
    responses = getResponse(sentiment_model, sentiment_tokenizer, text, params)
    return responses
  except Exception as e:
    logger.exception(e)
    return {"API Error": str(e)}