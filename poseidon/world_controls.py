"""Retain planner evidence around the unchanged, replayable TidePool engine."""
from __future__ import annotations

import copy

from .world import rollout


class RecordingController:
    """Call a planner once per action and keep detached decision evidence."""

    def __init__(self, controller):
        self.controller = controller
        self.decisions = []
        self.plan = getattr(controller, "plan", None)
        self.choose = controller.act if hasattr(controller, "act") else controller
        if not callable(self.choose) and not callable(self.plan):
            raise TypeError("policy must be callable or expose act(observation) or plan(observation)")

    def act(self, observation):
        if callable(self.plan):
            decision = self.plan(observation)
            if not isinstance(decision, dict) or "action" not in decision:
                raise ValueError("Planner must return a decision containing an action")
            action = decision["action"]
            self.decisions.append(copy.deepcopy(decision))
            return action
        self.decisions.append(None)
        return self.choose(observation)


def rollout_with_decisions(controller, seed=0, max_steps=256, scarcity=1.0):
    recorder = RecordingController(controller)
    episode = rollout(recorder, seed=seed, max_steps=max_steps, scarcity=scarcity)
    if len(recorder.decisions) != len(episode["trajectory"]):
        raise RuntimeError("Planner evidence and executed transitions disagree")
    for event, decision in zip(episode["trajectory"], recorder.decisions):
        if decision is not None:
            event["decision"] = decision
    return episode
