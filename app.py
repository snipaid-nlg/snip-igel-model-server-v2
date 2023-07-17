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
    # Get model and tokenizer from context
    model = context.get("model")
    tokenizer = context.get("tokenizer")

    # Get arguments from request
    prompt = request.json.get("prompt")
    document = request.json.get("document")
    task_prefix = request.json.get("task_prefix")
    params = request.json.get("params")

    # Handle missing arguments
    if document == None:
        return Response(
            json = {"message": "No document provided"}, 
            status=500
        )

    if task_prefix == None:
        task_prefix = ""

    if prompt == None:
        return Response(
            json = {"message": "No prompt provided"}, 
            status=500
        )
    
    if params == None:
        params = {}

    # Config
    generation_config = GenerationConfig(
        top_p=0.9,
        top_k=0,
        temperature=1,
        do_sample=True,
        early_stopping=True,
        length_penalty=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        **params
    )

    # Assemble prompt
    prompt = generate_prompt(prompt, document)

    # Embed prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # Generate
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )

    # Decode and return the result as a dictionary
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output_text = output.split("### Antwort:")[1].strip()
        return Response(
            json = {"output": output_text}, 
            status=200
        )
    
def generate_prompt(instruction, input=None) -> str:
    if input:
        return f"""Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt, und eine Eingabe, die weiteren Kontext liefert. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.

### Anweisung:
{instruction}

### Eingabe:
{input}

### Antwort:"""
    else:
        return f"""Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.

### Anweisung:
{instruction}

### Antwort:"""

if __name__ == "__main__":
    app.serve()