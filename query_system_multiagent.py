from __future__ import annotations
from typing import Any
from agents.a5 import (
    NLUnderstandingAgent,
    SecurityAgent,
    QueryPlannerAgent,
    QueryExecutionAgent,
    DiagnosisAgent,
    QueryRepairAgent,
    ExplanationAgent,
    ResponseGenerationAgent,
)

# Instantiate all agents
nlu_agent = NLUnderstandingAgent()
security_agent = SecurityAgent()
planner_agent = QueryPlannerAgent()
executor_agent = QueryExecutionAgent()
diagnosis_agent = DiagnosisAgent()
repair_agent = QueryRepairAgent()
explanation_agent = ExplanationAgent()
responder_agent = ResponseGenerationAgent()  # Now has internal validation


def answer_question(question: str) -> dict[str, Any]:
    """
    Orchestrates the multi-agent pipeline to answer a question,
    with a repair loop based on the final answer's quality.
    """
    # 1. NL Understanding
    intent = nlu_agent.run(question)

    # 2. Security Check
    security_check = security_agent.run(question)
    if security_check["decision"] == "REJECT":
        return {
            "answer": "Your request was blocked by the security agent.",
            "safety_decision": "REJECT",
            "diagnosis": "SECURITY_BLOCK",
            "repair_attempted": False,
            "repair_changed": False,
            "explanation": security_check["reason"],
        }

    # 3. Initial Query Plan & Execution
    plan = planner_agent.run(intent, question)
    execution = executor_agent.run(plan)

    # 4. First-pass Explanation and Response
    explanation = explanation_agent.run(execution)
    answer = responder_agent.run(question, explanation)

    # 5. Diagnose final answer and attempt repair if needed
    repair_attempted = False
    repair_changed = False

    # Check if the first answer indicates a lack of information
    # Only trigger repair if explicitly says "insufficient data"
    is_insufficient = answer.lower().strip() == "insufficient data"

    if is_insufficient:
        repair_attempted = True
        # The diagnosis for repair is now explicitly NO_DATA
        diagnosis_for_repair = {"label": "NO_DATA",
                                "reason": "Initial answer was insufficient."}
        repaired_plan = repair_agent.run(
            diagnosis_for_repair, intent, question)

        # Check if repair produced a valid plan with cypher queries
        if repaired_plan.get("cypher_typed") and repaired_plan.get("cypher_broad"):
            if repaired_plan["cypher_typed"] != plan.get("cypher_typed"):
                repair_changed = True
                # Re-run the pipeline with the new plan
                execution = executor_agent.run(repaired_plan)
                explanation = explanation_agent.run(execution)
                answer = responder_agent.run(question, explanation)

    # 6. Final Diagnosis based on the final execution result
    final_diagnosis_obj = diagnosis_agent.run(execution)
    final_diagnosis_label = final_diagnosis_obj["label"]
    if execution.get("error"):
        final_diagnosis_label = "QUERY_ERROR"
    # If we repaired due to insufficient answer, but still got no data, reflect that.
    elif repair_changed and not execution.get("rows"):
        final_diagnosis_label = "NO_DATA"

    return {
        "answer": answer,
        "safety_decision": "ALLOW",
        "diagnosis": final_diagnosis_label,
        "repair_attempted": repair_attempted,
        "repair_changed": repair_changed,
        "explanation": explanation,
    }


def run_multiagent_qa(question: str) -> dict[str, Any]:
    return answer_question(question)


if __name__ == "__main__":
    while True:
        q = input("Question (type exit): ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print(answer_question(q))
