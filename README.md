# Priority Task Agent 🎯

## Problem
Every day, students and working professionals face the same question:
**"What should I do first?"**

Wrong prioritization leads to missed deadlines, stress, and burnout.
This RL environment trains an AI agent to make smarter daily decisions.

## What This Environment Does
- Agent receives 5 tasks with deadlines, importance, and effort levels
- Agent decides which task to do first
- Correct prioritization = reward, wrong = penalty
- Agent learns the optimal sequence over time

## Reward System
| Choice | Reward |
|--------|--------|
| Best task first | +2.0 |
| Good choice | +1.0 |
| Okay choice | 0.0 |
| Poor choice | -1.0 |
| Missed deadline | -2.0 |

## How to Run
```python
from priority_task_env import PriorityTaskEnv

env = PriorityTaskEnv()
tasks = env.reset()
result = env.step(task_id=0)
print(result)
```

## Built With
- Python
- OpenEnv framework
- Meta x PyTorch x SST Hackathon 2026

