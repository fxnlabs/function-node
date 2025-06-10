package registry

import "time"

func NewNodeRegistry(interval time.Duration) *CachedRegistry {
	return NewCachedRegistry(fetchNodeRegistry, interval)
}

func fetchNodeRegistry() (map[string]interface{}, error) {
	// TODO: Implement actual smart contract call to fetch node registry
	// For now, returning a dummy implementation
	return map[string]interface{}{
		"node1": "address1",
		"node2": "address2",
	}, nil
}
