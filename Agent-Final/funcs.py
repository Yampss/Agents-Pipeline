from typing import TypedDict
from langgraph.graph import StateGraph, END
from utility_functions import transcribe_folder_to_csv, process_folder_vad, save_num_speakers

folder_path = "/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-202154810"

class StateDict(TypedDict, total=False):
    A: str  # Transcription path
    D: str  # Silence CSV path
    E: str  # Speaker CSV path

def A_func(state: StateDict) -> StateDict:
    print("Running Transcription")
    result = transcribe_folder_to_csv(folder_path, source_language="Hindi")  # or "Marathi"
    return {"A": result}

def D_func(state: StateDict) -> StateDict:
    print("Running Silence Detection")
    result = process_folder_vad(folder_path)
    return {"D": result}

def E_func(state: StateDict) -> StateDict:
    print("Running Speaker Diarization")
    result = save_num_speakers(folder_path)
    return {"E": result}

node_map = {
    1: ("node_A", A_func, "A"),
    4: ("node_D", D_func, "D"),
    5: ("node_E", E_func, "E")
}

# Dynamic graph builder
def build_graph_from_structure(structure: list[list[int]]):
    graph_builder = StateGraph(StateDict)

    for group in structure:
        for node_id in group:
            node_name, func, _ = node_map[node_id]
            graph_builder.add_node(node_name, func)

    for i in range(len(structure) - 1):
        for src in structure[i]:
            for dst in structure[i + 1]:
                graph_builder.add_edge(node_map[src][0], node_map[dst][0])

    graph_builder.set_entry_point(node_map[structure[0][0]][0])

    for node_id in structure[-1]:
        graph_builder.add_edge(node_map[node_id][0], END)

    return graph_builder.compile()

structure = [[1], [4, 5]] 
graph = build_graph_from_structure(structure)
graph.get_graph().print_ascii()

# Invoke the graph
initial_state = {}
final_result = graph.invoke(initial_state)
print("Final result:", final_result)
