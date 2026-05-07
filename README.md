Architecture diagram, 
                +----------------------+
                |   Regulation Files   |
                +----------------------+
                           |
                           v
                +----------------------+
                |     build_kg.py      |
                |  KG Construction     |
                +----------------------+
                           |
                           v
                +----------------------+
                |   Neo4j KG Database  |
                +----------------------+
                           ^
                           |
         +--------------------------------------+
         |         Multi-Agent QA System        |
         |                                      |
         |  - NLUnderstandingAgent              |
         |  - SecurityAgent                     |
         |  - QueryPlannerAgent                 |
         |  - QueryExecutionAgent               |
         |  - ExplanationAgent                  |
         |  - ResponseGenerationAgent           |
         |  - ValidationAgent                   |
         |  - RepairAgent                       |
         +--------------------------------------+
                           ^
                           |
                    +-------------+
                    | User Query  |
                    +-------------+

         +----------------------+
         |      Local LLM       |
         +----------------------+
                   ^
                   |
      +-----------------------------+
      | Security / Response Agents  |
      +-----------------------------+
agent responsibilities, 
此處agent 大致與template 相同 
nlu 透過regex keywrod extraction 等方式 先拆解用戶問提供後續使用
security 在接收到惡意指令時會直接停止整個agent_flow
queryplanner 則會在接收到nlu的intent後 透過llm 和
pipeline, 

User Question
      |
      v
+----------------------+
| SecurityAgent        |
+----------------------+
      |
   +--+----------------------+
   |                         |
   | REJECT                 | ALLOW
   |                         |
   v                         v
BLOCKED               NLUnderstandingAgent
                            |
                            v
|-------------------->QueryPlannerAgent
|                            |
|                            v
|                 QueryExecutionAgent
|                            |
|                            v
|                   ExplanationAgent
|                            |
|                            v
|             ResponseGenerationAgent
|                            |
|                            v
|             ResponseValidationAgent
|                            |
|                 +----------+----------+
|                 |                     |
|         SUCCESS (final)      FAILURE / insufficient
|                 |                     |
|                 v                     v
|            Final Answer      QueryRepairAgent(it will only do once.)
|                                      |
|                                      v
|                 +--------------------+--------------------+
|                 |                                         |
|                 v                                         v
+-------QueryPlannerAgent                         QueryExecutionAgent
        (re-plan strategy)                        (direct re-run)
challenges,
 findings