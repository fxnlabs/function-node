package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	SchedulerChecks = promauto.NewCounter(prometheus.CounterOpts{
		Name: "scheduler_checks_total",
		Help: "The total number of scheduler checks",
	})

	EndpointResponses = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "endpoint_responses_total",
		Help: "The total number of endpoint responses",
	}, []string{"endpoint", "status_code"})
)
