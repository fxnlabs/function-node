package challengers

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"
)

// EndpointReachableChallenger checks if an endpoint is reachable.
type EndpointReachableChallenger struct{}

// Execute checks if an endpoint is reachable.
func (c *EndpointReachableChallenger) Execute(payload interface{}, log *zap.Logger) (interface{}, error) {
	endpoint, ok := payload.(string)
	if !ok {
		return nil, fmt.Errorf("invalid payload for endpoint reachability challenge")
	}

	log.Info("Polling endpoint", zap.String("endpoint", endpoint))

	resp, err := http.Get(endpoint)
	if err != nil {
		log.Error("Failed to reach endpoint", zap.String("endpoint", endpoint), zap.Error(err))
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		log.Info("Endpoint is reachable", zap.String("endpoint", endpoint), zap.Int("status_code", resp.StatusCode))
		return map[string]bool{"reachable": true}, nil
	}

	log.Warn("Endpoint is not reachable", zap.String("endpoint", endpoint), zap.Int("status_code", resp.StatusCode))
	return map[string]bool{"reachable": false}, nil
}
