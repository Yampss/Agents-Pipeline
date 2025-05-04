# 


from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
import torch
import librosa
import nemo.collections.asr as nemo_asr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hi_indicconf_model_path = '/root/abhinav/models/indicconformer_stt_hi_hybrid_rnnt_large.nemo'
hi_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=hi_indicconf_model_path)
hi_indicconf_model.freeze()
hi_indicconf_model = hi_indicconf_model.to(device)
hi_indicconf_model.cur_decoder = "ctc"

def transcribe_audio_tool(input: dict) -> str:
    """
    Expects input like:
    {
        "audio_path": "/path/to/audio.wav",
        "source_lang": "Hindi"
    }
    """
    try:
        audio_path = input["audio_path"]
        source_lang = input.get("source_lang", "Hindi") 
        
        if source_lang == "Hindi":
            print(f"Transcribing {audio_path} in Hindi...")
            transcription = hi_indicconf_model.transcribe(
                [audio_path], batch_size=1, logprobs=False, language_id="hi"
            )
            if transcription and len(transcription) > 0:
                words = transcription[0].strip().split()
                return " ".join(words)
            else:
                return "Transcription returned empty result"
        else:
            return f"Unsupported language: {source_lang}"
    except Exception as e:
        return f"Error in transcription: {e}"

transcribe_tool = Tool.from_function(
    func=transcribe_audio_tool,
    name="transcribe_audio",
    description="Transcribes Hindi speech audio to text. Input must be a dictionary with 'audio_path' and 'source_lang' (optional, defaults to Hindi)."
)

toolbox = [transcribe_tool]

llm = ChatOllama(model="llama3.1:latest", temperature=0)

# def assistant_node(state):
#     messages = state["messages"]
#     # Explicitly configure the LLM to use tools
#     response = llm.invoke(messages, tools=toolbox)
#     return {"messages": messages + [response]}
def assistant_node(state):
    messages = state["messages"]
    # Pass the function(s) directly, not Tool objects
    response = llm.invoke(messages, tools=[transcribe_adio_tool])
    return {"messages": messages + [response]}
# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("assistant", assistant_node)
graph_builder.add_node("tools", ToolNode(toolbox))
graph_builder.set_entry_point("assistant")
graph_builder.add_conditional_edges(
    "assistant", 
    tools_condition,
    {
        "tools": "tools",
        "assistant": "assistant"
    }
)
graph_builder.add_edge("tools", "assistant")
graph = graph_builder.compile()

inputs = {
    "messages": [
        HumanMessage(content="Use the transcribe_audio tool and transcribe this Hindi audio file: /root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-201932960/sent_21.wav"),
    ]
}
for output in graph.stream(inputs):
    if "assistant" in output:
        messages = output["assistant"]["messages"]
        if len(messages) > 1:  
            print("\n=== New Message ===")
            print(messages[-1].content)
    elif "tools" in output:
        print("\n=== Tool Execution ===")
        print(output["tools"]["messages"][-1].content)