from app.schemas.contracts import Decision


class DecisionPolicy:
    """Per-session decision state for future real inference policies."""

    def __init__(self) -> None:
        self.last_spoken_prediction_id: str | None = None

    def previous_for(self, decision: Decision) -> str | None:
        if decision == Decision.UPDATE:
            return self.last_spoken_prediction_id
        return None

    def record(self, decision: Decision, prediction_id: str) -> None:
        if decision in {Decision.SPEAK, Decision.UPDATE}:
            self.last_spoken_prediction_id = prediction_id

