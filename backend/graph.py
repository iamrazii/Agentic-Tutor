
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.tutor_core import TutorState, topic_exists, topic_selection,overview_node,evaluation_mode,MCQFormation,SQFormation,ConductingTestMCQ,ConductingTestSQ

from backend.tutor_core import user_queries




# Graph setup
graph = StateGraph(TutorState)

# Settign up nodes

graph.add_node("select_topic", topic_selection )
graph.add_node("overview", overview_node , checkpoint =True)
graph.add_node("evaluation_mode" , evaluation_mode,checkpoint=True)
graph.add_node("MCQFormation",MCQFormation)
graph.add_node("SQFormation" , SQFormation)
graph.add_node("ConductingTestMCQ",ConductingTestMCQ)
graph.add_node("ConductingTestSQ",ConductingTestSQ)

# Settign up edges / flow

graph.add_edge(START,'select_topic')
graph.add_conditional_edges(
    "select_topic",
    lambda state: state.get("branch", "invalid"),  # branch key
    {
        "valid": "overview",
        "invalid": END,  # loop back
        "awaiting_topic": END
    },
)
graph.add_edge('overview',"evaluation_mode")
graph.add_conditional_edges(
    "evaluation_mode",
    lambda state: state.get("evaluation_style") if state.get("evaluation_style") else "wait",
    {
        "MCQ": "MCQFormation",
        "SQ": "SQFormation",
        "wait" : END
    }
)


def route_to_test(state):
    # If there is a user answer, route to the respective test grader
    if state.get("user_answer"):
        return "ConductingTestMCQ" if state["evaluation_style"] == "MCQ" else "ConductingTestSQ"
    # Otherwise, go to sleep (END) and wait for the user to answer
    return END

graph.add_conditional_edges("MCQFormation", route_to_test)
graph.add_conditional_edges("SQFormation", route_to_test)

# Loop the test graders back to END so they wait for the next question
graph.add_edge("ConductingTestMCQ", END)
graph.add_edge("ConductingTestSQ", END)


memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)