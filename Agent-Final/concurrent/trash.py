from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
import torch
import librosa
import nemo.collections.asr as nemo_asr
import os
from dotenv import load_dotenv

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hi_indicconf_model_path = '/root/abhinav/models/indicconformer_stt_hi_hybrid_rnnt_large.nemo'
hi_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=hi_indicconf_model_path)
hi_indicconf_model.freeze()
hi_indicconf_model = hi_indicconf_model.to(device)
hi_indicconf_model.cur_decoder = "ctc"

def transcribe_audio_tool(input: dict) -> str:
    
    try:
        audio_path = input.get("audio_path")
        if not audio_path:
            return "Error: No audio_path provided"
            
        source_lang = input.get("source_lang", "Hindi")
        
        if source_lang.lower() == "hindi":
            print(f"Transcribing {audio_path} in Hindi...")
            transcription = hi_indicconf_model.transcribe(
                [audio_path], batch_size=1, logprobs=False, language_id="hi"
            )
            if transcription and len(transcription) > 0:
                result = transcription[0].strip()
                print(f"Transcription result: {result}")
                return result
            else:
                return "Transcription returned empty result"
        else:
            return f"Unsupported language: {source_lang}. Currently only Hindi is supported."
    except Exception as e:
        return f"Error in transcription: {str(e)}"

transcribe_tool = Tool.from_function(
    func=transcribe_audio_tool,
    name="transcribe_audio",
    description="Transcribes Hindi speech audio to text. Input must be a dictionary with 'audio_path' (required) and 'source_lang' (optional, defaults to Hindi)."
)

toolbox = [transcribe_tool]

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

def assistant_node(state):
    messages = state["messages"]
    response = llm.invoke(messages, tools=toolbox)
    return {"messages": messages + [response]}

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

print("Starting conversation...")
for output in graph.stream(inputs):
    if "assistant" in output:
        messages = output["assistant"]["messages"]
        if len(messages) > 1:
            print("\n=== New Assistant Message ===")
            print(messages[-1].content)
    elif "tools" in output:
        print("\n=== Tool Execution ===")
        print(output["tools"]["messages"][-1].content)

print("\nFinal conversation:")
for i, msg in enumerate(graph.invoke(inputs)["messages"]):
    role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
    print(f"\n{role}: {msg.content}")