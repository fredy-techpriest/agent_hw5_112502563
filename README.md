Architecture diagram, 
```text
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
|        Multi-Agent QA System         |
|                                      |
| - NLUnderstandingAgent              |
| - SecurityAgent                     |
| - QueryPlannerAgent                |
| - QueryExecutionAgent              |
| - ExplanationAgent                 |
| - ResponseGenerationAgent          |
| - ValidationAgent                  |
| - RepairAgent                      |
+--------------------------------------+
           ^
           |
     +-------------+
     | User Query  |
     +-------------+
```
agent responsibilities, 
此處agent 大致與template 相同 
nlu 透過regex keywrod extraction 等方式 先拆解用戶問提供後續使用
security 在接收到惡意指令時會直接停止整個agent_flow
queryplanner 則會在接收到nlu的intent後 透過llm 和
pipeline, 
```text
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
```

challenges,
由於底層的kg graph 並不完善，因此在面對正常的task 時分數較低，除此之外我在response agent 遇到的問題是 我在prompt 中要求他在資料不足時輸出insufficient data 
但此舉不知為何儘管有足夠的資料 仍然十分容易回傳insufficient data 因此只好另創一個validator 應對此情況
 findings
 
 此處findings 就是我應該要在上一份作業就加入使用embedding 等方法 可以有效地加強我的query agent 的做法 
 同時我現在看才覺得build_kg 將規則切得有些太細因此不方便後續實作
 同時也注意到可能對於這個問題而言模型太小 無法很正確的直接判斷資訊量是否足夠
