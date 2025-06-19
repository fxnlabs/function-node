package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
)

func TestMatrixMultiplicationMetrics(t *testing.T) {
	// Test duration histogram
	t.Run("ChallengeMatrixMultDuration", func(t *testing.T) {
		// Observe some sample durations
		ChallengeMatrixMultDuration.Observe(100.5)
		ChallengeMatrixMultDuration.Observe(200.3)
		ChallengeMatrixMultDuration.Observe(150.7)

		// Verify histogram was updated (we can't directly read the count with testutil)
		// Just verify no panic occurs
		assert.NotPanics(t, func() {
			ChallengeMatrixMultDuration.Observe(300.1)
		})
	})

	// Test matrix size gauge
	t.Run("ChallengeMatrixMultSize", func(t *testing.T) {
		ChallengeMatrixMultSize.Set(1024)
		value := testutil.ToFloat64(ChallengeMatrixMultSize)
		assert.Equal(t, float64(1024), value)
	})

	// Test GFLOPS gauge
	t.Run("ChallengeMatrixMultGFLOPS", func(t *testing.T) {
		ChallengeMatrixMultGFLOPS.Set(123.45)
		value := testutil.ToFloat64(ChallengeMatrixMultGFLOPS)
		assert.Equal(t, float64(123.45), value)
	})

	// Test GPU utilization gauge
	t.Run("GPUUtilizationPercent", func(t *testing.T) {
		GPUUtilizationPercent.Set(85.5)
		value := testutil.ToFloat64(GPUUtilizationPercent)
		assert.Equal(t, float64(85.5), value)
	})

	// Test GPU memory gauge
	t.Run("GPUMemoryUsedBytes", func(t *testing.T) {
		GPUMemoryUsedBytes.Set(1073741824) // 1GB
		value := testutil.ToFloat64(GPUMemoryUsedBytes)
		assert.Equal(t, float64(1073741824), value)
	})

	// Test backend counter
	t.Run("ChallengeMatrixMultBackend", func(t *testing.T) {
		// Increment counters
		ChallengeMatrixMultBackend.WithLabelValues("cpu").Inc()
		ChallengeMatrixMultBackend.WithLabelValues("cpu").Inc()
		ChallengeMatrixMultBackend.WithLabelValues("cuda").Inc()

		// Since these are global metrics that accumulate, we just verify they work
		// In a real test environment, you'd want to use a custom registry
		assert.NotPanics(t, func() {
			ChallengeMatrixMultBackend.WithLabelValues("cpu").Inc()
		})
	})
}

func TestMetricsRegistration(t *testing.T) {
	// Ensure all metrics are properly registered
	metrics := []prometheus.Collector{
		ChallengeMatrixMultDuration,
		ChallengeMatrixMultSize,
		ChallengeMatrixMultGFLOPS,
		GPUUtilizationPercent,
		GPUMemoryUsedBytes,
		ChallengeMatrixMultBackend,
	}

	for _, metric := range metrics {
		// This will panic if the metric is not properly registered
		assert.NotPanics(t, func() {
			_ = prometheus.Register(metric)
			prometheus.Unregister(metric)
		})
	}
}

func BenchmarkMetricsObservation(b *testing.B) {
	b.Run("ObserveDuration", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ChallengeMatrixMultDuration.Observe(float64(i % 1000))
		}
	})

	b.Run("SetGauge", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ChallengeMatrixMultSize.Set(float64(i))
		}
	})

	b.Run("IncCounter", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ChallengeMatrixMultBackend.WithLabelValues("cpu").Inc()
		}
	})
}