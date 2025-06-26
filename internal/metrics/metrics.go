package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	EndpointResponses = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "endpoint_responses_total",
		Help: "The total number of endpoint responses",
	}, []string{"endpoint", "status_code"})

	// Matrix Multiplication Challenge Metrics
	ChallengeMatrixMultDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "challenge_matrix_mult_duration_ms",
		Help:    "Duration of matrix multiplication computation in milliseconds",
		Buckets: prometheus.ExponentialBuckets(1, 2, 15), // 1ms to ~32s
	})

	ChallengeMatrixMultSize = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "challenge_matrix_mult_size",
		Help: "Size of the matrix used in the last multiplication challenge",
	})

	ChallengeMatrixMultGFLOPS = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "challenge_matrix_mult_gflops",
		Help: "Performance of the last matrix multiplication in GFLOPS",
	})

	ChallengeMatrixMultBackend = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "challenge_matrix_mult_backend_total",
		Help: "Total number of matrix multiplication challenges by backend",
	}, []string{"backend"})

	// GPU Metrics
	GPUUtilizationPercent = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gpu_utilization_percent",
		Help: "Current GPU utilization percentage (0-100)",
	})

	GPUMemoryUsedBytes = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gpu_memory_used_bytes",
		Help: "GPU memory currently in use in bytes",
	})
)
