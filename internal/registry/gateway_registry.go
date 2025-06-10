package registry

import "time"

var GatewayRegistry *CachedRegistry

func NewGatewayRegistry(interval time.Duration) *CachedRegistry {
	return NewCachedRegistry(fetchGatewayRegistry, interval)
}

func fetchGatewayRegistry() (map[string]interface{}, error) {
	// TODO: Implement actual smart contract call to fetch gateway registry
	// For now, returning a dummy implementation
	return map[string]interface{}{
		"gateway1": "address1",
		"gateway2": "address2",
	}, nil
}
