*LLM-IE* Python package and web application installation guide.

## Python package
The Python package is available on PyPI. 

```bash
pip install llm-ie 
```
Note that this package does not check LLM inference engine installation nor install them. At least one LLM inference engine is required. There are built-in supports for [LiteLLM](https://github.com/BerriAI/litellm), [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [Ollama](https://github.com/ollama/ollama), [Huggingface_hub](https://github.com/huggingface/huggingface_hub), [OpenAI API](https://platform.openai.com/docs/api-reference/introduction), and [vLLM](https://github.com/vllm-project/vllm). For installation guides, please refer to those projects. Other inference engines can be configured through the [InferenceEngine](src/llm_ie/engines.py) abstract class. See [LLM Inference Engine](./llm_inference_engine.md) section.

## Web Application
### Docker
The easiest way to install our web application is 🐳Docker. The image is available on Docker Hub. Use the command below to pull and run locally:
```bash
docker pull daviden1013/llm-ie-web-app:latest
docker run -p 5000:5000 daviden1013/llm-ie-web-app:latest
```

Open your web browser and navigate to: http://localhost:5000

If port 5000 is already in use on your machine, you can map it to a different local port. For example, to map it to local port 8080:
```bash
docker run -p 8080:5000 daviden1013/llm-ie-web-app:latest
```

Then visit http://localhost:8080

### Install from source
Alternatively, pull the repo and build the required environment locally.

```bash
# Clone source code
git clone https://github.com/daviden1013/llm-ie.git

# Install requirements
pip install -r llm-ie/web_app/requirements.txt

# Run Web App
cd llm-ie/web_app
python run.py
```