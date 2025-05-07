Steps to use:
dont forgt to use CUDA_VISIBLE_DEVIVCES to attach to single GPU
Open 3 terminals in container or server and run commands:

terminal 1 :
ollama serve

terimnal 2:
ollama run llama3.3:70b

terminal 3:
python llm.py
