"""
monitoring/metrics.py

Minimal in-process metrics collector for the API. In a real deployment
this would be replaced by / wired into Prometheus + Grafana, but this
gives a working baseline that satisfies the monitoring requirement:
  - API: request count, latency, errors, availability
  - ML: prediction distribution, confidence
"""

import time
from collections import defaultdict, deque
from statistics import mean

WINDOW = 500  # keep the last N predictions for rolling stats


class MetricsStore:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latencies = deque(maxlen=WINDOW)
        self.predicted_classes = deque(maxlen=WINDOW)
        self.confidences = deque(maxlen=WINDOW)
        self.start_time = time.time()

    def record_request(self, latency_s: float, error: bool = False):
        self.request_count += 1
        if error:
            self.error_count += 1
        self.latencies.append(latency_s)

    def record_prediction(self, predicted_class: str, confidence: float):
        self.predicted_classes.append(predicted_class)
        self.confidences.append(confidence)

    def snapshot(self) -> dict:
        uptime = time.time() - self.start_time
        class_counts = defaultdict(int)
        for c in self.predicted_classes:
            class_counts[c] += 1

        return {
            "uptime_seconds": round(uptime, 1),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_count / self.request_count, 4) if self.request_count else 0.0,
            "avg_latency_ms": round(mean(self.latencies) * 1000, 2) if self.latencies else None,
            "prediction_distribution": dict(class_counts),
            "avg_confidence": round(mean(self.confidences), 4) if self.confidences else None,
        }


metrics_store = MetricsStore()
