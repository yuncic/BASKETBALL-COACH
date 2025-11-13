from typing import Optional, List
from pydantic import BaseModel


class MetricsKneeHip(BaseModel):
    gap: str
    verdict: str


class MetricsShoulderElbow(BaseModel):
    gap: str
    verdict: str


class MetricsReleaseTiming(BaseModel):
    gap: str
    verdict: str


class Metrics(BaseModel):
    knee_hip: MetricsKneeHip
    shoulder_elbow: MetricsShoulderElbow
    release_timing: MetricsReleaseTiming


class Alignment(BaseModel):
    arm_ball: float
    com_ball: float
    release_angle: float


class Report(BaseModel):
    eff_score: float
    metrics: Metrics
    alignment: Alignment
    suggestions: List[str]

