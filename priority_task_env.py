"""
Priority Task Agent Environment
================================
Hackathon: Meta x PyTorch x SST — Round 1
Builder: Solo submission

Problem Statement:
------------------
Students and working professionals struggle daily with deciding WHAT to do first.
Given a list of tasks with deadlines, importance, and effort levels,
an AI agent must learn to prioritize them correctly.

Wrong prioritization = missed deadlines = real-world consequence.
This environment trains an agent to make better daily decisions.

OpenEnv Interface: reset() → step() → state()
"""

import random
from dataclasses import dataclass
from typing import Optional


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Task:
    """A single task in the agent's inbox."""
    id: int
    name: str
    deadline_hours: float   # hours remaining until due
    importance: int         # 1 (low) to 5 (high)
    effort_hours: float     # estimated hours to complete


@dataclass
class PrioritizeAction:
    """Agent's action: pick which task to work on next."""
    task_id: int


@dataclass
class Observation:
    """What the agent sees each step."""
    tasks: list[dict]           # list of task dicts
    completed_tasks: list[int]  # IDs of tasks already done
    current_time: float         # hours elapsed
    step_number: int


@dataclass
class StepResult:
    """Result returned after each step."""
    observation: Observation
    reward: float
    done: bool
    info: dict


# ── Task Generator ────────────────────────────────────────────────────────────

TASK_TEMPLATES = [
    {"name": "Submit assignment",     "importance": 5, "effort_hours": 2.0},
    {"name": "Reply to urgent email", "importance": 4, "effort_hours": 0.5},
    {"name": "Attend team meeting",   "importance": 5, "effort_hours": 1.0},
    {"name": "Read study notes",      "importance": 3, "effort_hours": 1.5},
    {"name": "Buy groceries",         "importance": 2, "effort_hours": 1.0},
    {"name": "Exercise",              "importance": 3, "effort_hours": 1.0},
    {"name": "Watch lecture video",   "importance": 4, "effort_hours": 2.0},
    {"name": "Clean room",            "importance": 1, "effort_hours": 1.5},
    {"name": "Prepare presentation",  "importance": 5, "effort_hours": 3.0},
    {"name": "Check notifications",   "importance": 1, "effort_hours": 0.25},
]

def generate_tasks(n: int = 5, seed: Optional[int] = None) -> list[Task]:
    """Generate a random set of tasks with deadlines."""
    if seed is not None:
        random.seed(seed)
    templates = random.sample(TASK_TEMPLATES, min(n, len(TASK_TEMPLATES)))
    tasks = []
    for i, t in enumerate(templates):
        deadline = round(random.uniform(1.0, 12.0), 1)  # 1 to 12 hours from now
        tasks.append(Task(
            id=i,
            name=t["name"],
            deadline_hours=deadline,
            importance=t["importance"],
            effort_hours=t["effort_hours"],
        ))
    return tasks


# ── Scoring Logic ─────────────────────────────────────────────────────────────

def compute_priority_score(task: Task) -> float:
    """
    Ground-truth score: what a smart human would rate this task.
    Higher = should be done sooner.
    
    Formula:
      - Urgency: high if deadline is close relative to effort needed
      - Importance: direct multiplier
      - Slack: how much buffer time exists
    """
    slack = task.deadline_hours - task.effort_hours
    urgency = 1.0 / max(slack, 0.1)       # less slack = more urgent
    score = urgency * task.importance
    return round(score, 4)


# ── Environment ───────────────────────────────────────────────────────────────

class PriorityTaskEnv:
    """
    OpenEnv-compatible RL environment for task prioritization.
    
    Episode flow:
    1. reset()  → agent gets a list of tasks
    2. step(action) → agent picks one task to do; gets reward + new observation
    3. Repeat until all tasks done or time runs out
    4. state()  → inspect current env state anytime
    
    Reward design:
    +2.0  → chose the highest priority task (optimal)
    +1.0  → chose a high priority task (within top 2)
     0.0  → chose a medium priority task
    -1.0  → chose a low priority task when urgent ones exist
    -2.0  → task missed deadline (overdue when picked)
    """

    MAX_STEPS = 10
    DAILY_HOURS = 10.0  # agent has 10 working hours

    def __init__(self, n_tasks: int = 5, seed: Optional[int] = None):
        self.n_tasks = n_tasks
        self.seed = seed
        self._tasks: list[Task] = []
        self._completed: list[int] = []
        self._current_time: float = 0.0
        self._step_num: int = 0
        self._done: bool = False

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode. Returns initial observation."""
        self._tasks = generate_tasks(self.n_tasks, self.seed)
        self._completed = []
        self._current_time = 0.0
        self._step_num = 0
        self._done = False
        return self._get_observation()

    def step(self, action: PrioritizeAction) -> StepResult:
        """
        Agent picks a task_id to work on.
        Returns reward + next observation + done flag.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        task_id = action.task_id
        pending = self._get_pending_tasks()

        # ── Validate action ────────────────────────────────────────────────
        valid_ids = [t.id for t in pending]
        if task_id not in valid_ids:
            return StepResult(
                observation=self._get_observation(),
                reward=-1.0,
                done=False,
                info={"error": f"Invalid task_id {task_id}. Valid: {valid_ids}"}
            )

        chosen_task = next(t for t in pending if t.id == task_id)

        # ── Check if deadline already missed ──────────────────────────────
        if self._current_time > chosen_task.deadline_hours:
            reward = -2.0
            info = {"result": "missed_deadline", "task": chosen_task.name}
        else:
            # ── Rank all pending tasks by priority score ───────────────────
            scored = sorted(pending, key=lambda t: compute_priority_score(t), reverse=True)
            rank = next(i for i, t in enumerate(scored) if t.id == task_id)

            if rank == 0:
                reward = 2.0
                info = {"result": "optimal_choice"}
            elif rank == 1:
                reward = 1.0
                info = {"result": "good_choice"}
            elif rank <= len(pending) // 2:
                reward = 0.0
                info = {"result": "acceptable_choice"}
            else:
                reward = -1.0
                info = {"result": "poor_choice"}

        # ── Update state ──────────────────────────────────────────────────
        self._completed.append(task_id)
        self._current_time += chosen_task.effort_hours
        self._step_num += 1

        # Episode ends when all tasks done or time/steps exhausted
        pending_after = self._get_pending_tasks()
        self._done = (
            len(pending_after) == 0
            or self._step_num >= self.MAX_STEPS
            or self._current_time >= self.DAILY_HOURS
        )

        info["task_done"] = chosen_task.name
        info["time_elapsed"] = round(self._current_time, 2)
        info["tasks_remaining"] = len(pending_after)

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self._done,
            info=info
        )

    def state(self) -> dict:
        """Return full current state (for debugging / evaluation)."""
        return {
            "step": self._step_num,
            "current_time_hours": round(self._current_time, 2),
            "completed_task_ids": self._completed,
            "pending_tasks": [self._task_to_dict(t) for t in self._get_pending_tasks()],
            "done": self._done,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_pending_tasks(self) -> list[Task]:
        return [t for t in self._tasks if t.id not in self._completed]

    def _get_observation(self) -> Observation:
        return Observation(
            tasks=[self._task_to_dict(t) for t in self._get_pending_tasks()],
            completed_tasks=list(self._completed),
            current_time=round(self._current_time, 2),
            step_number=self._step_num,
        )

    def _task_to_dict(self, task: Task) -> dict:
        return {
            "id": task.id,
            "name": task.name,
            "deadline_hours": task.deadline_hours,
            "importance": task.importance,
            "effort_hours": task.effort_hours,
            "priority_score": compute_priority_score(task),
        }


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Priority Task Agent — Environment Test Run")
    print("=" * 55)

    env = PriorityTaskEnv(n_tasks=5, seed=42)
    obs = env.reset()

    print("\nInitial Tasks:")
    for t in obs.tasks:
        print(f"  [{t['id']}] {t['name']:<28} | deadline: {t['deadline_hours']}h "
              f"| importance: {t['importance']} | score: {t['priority_score']}")

    total_reward = 0.0
    step = 0

    while True:
        # Simple greedy agent: always picks highest priority_score task
        pending = obs.tasks
        if not pending:
            break
        best = max(pending, key=lambda t: t["priority_score"])
        action = PrioritizeAction(task_id=best["id"])

        result = env.step(action)
        total_reward += result.reward
        step += 1

        print(f"\nStep {step}: Chose '{best['name']}'")
        print(f"  Reward: {result.reward:+.1f} | {result.info.get('result','')}")
        print(f"  Time elapsed: {result.info['time_elapsed']}h | "
              f"Tasks remaining: {result.info['tasks_remaining']}")

        obs = result.observation
        if result.done:
            break

    print(f"\n{'=' * 55}")
    print(f"  Episode finished! Total reward: {total_reward:+.1f}")
    print(f"  Final state: {env.state()}")
    print("=" * 55)
