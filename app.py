from potassium import Potassium, Request, Response

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    # load model
    print("loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "malteos/bloom-6b4-clp-german",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto'
    )
    print("done")
    
    # load LoRA adapters finetuned for news snippet generation
    print("downloading adapter checkpoint")
    PeftModel.from_pretrained(model, "snipaid/snip-igel-500-v2")
    print("done")

    # load tokenizer
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("philschmid/instruct-igel-001")
    print("done")
   
    context = {
        "model": model,
        "tokenizer": tokenizer
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()