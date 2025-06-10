package challenge

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/fxnlabs/function-node/internal/challenge/challengers"
	"go.uber.org/zap"
)

// Challenger defines the interface for a challenge.
type Challenger interface {
	Execute(payload interface{}, log *zap.Logger) (interface{}, error)
}

// Challenge represents a challenge from the Scheduler.
type Challenge struct {
	Type    string      `json:"type"`
	Payload interface{} `json:"payload"`
}

// NewChallenger creates a new challenger based on the challenge type.
func NewChallenger(challengeType string) (Challenger, error) {
	switch challengeType {
	case "POLL_GPU_STATS":
		return &challengers.GPUStatsChallenger{}, nil
	case "MATRIX_MULTIPLICATION":
		return &challengers.MatrixMultiplicationChallenger{}, nil
	case "POLL_ENDPOINT_REACHABLE":
		return &challengers.EndpointReachableChallenger{}, nil
	default:
		return nil, fmt.Errorf("unknown challenge type: %s", challengeType)
	}
}

// ChallengeHandler handles challenge requests.
func ChallengeHandler(log *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var challenge Challenge
		if err := json.NewDecoder(r.Body).Decode(&challenge); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		challenger, err := NewChallenger(challenge.Type)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		result, err := challenger.Execute(challenge.Payload, log)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}
