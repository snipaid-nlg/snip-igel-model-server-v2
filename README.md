# SNIP-IGEL Potassium Model Server
This is a Potassium HTTP server for our SNIP-IGEL model finetuned for news snippet generation.

## Quickstart

> _**Note:** This model requires a GPU with ~ 12GB memory for generation!_

Curious to get your hand on SNIP-IGEL?

You can check it out with docker:

1. Run `docker build -t snip-igel-500-v2 . && docker run -it snip-igel-500-v2` to build and run the docker container.

Or you can check it out manually:

1. Run `pip3 install -r requirements.txt` to download dependencies.
2. Run `python3 app.py` to start the server.
3. You should see:

```
------  
Starting server 🍌  
Running init()  
...  
Serving on http://localhost:8000  
------
```

4. Now open up a different terminal and hit the server with a simple cURL POST request

```
curl -X POST \
-H "Content-Type: application/json" \
-d '{prompt": "Generate a title for the following news article.", "document": "<Insert-the-fulltext-of-a-news-article-here>"}' \
http://localhost:8000/
```
5. Boom! 🎉 You just ran an inference on the model on your local machine!
```
{
    "output": "Here is the text the model generated."
}
```

## 🍌
## Test and deploy with Banana

### Testing

> _**Note:** For this you need the banana-cli installed. Run `pip3 install banana-cli` to install it._

1. Fork this repo and clone it to your local device.
2. Start a local dev server with `banana dev`.

### Deployment
1. [Log into Banana](https://app.banana.dev/onboard).
4. Select your fork of the repo to build and deploy!
