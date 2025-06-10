package registry

import "time"

var SchedulerRegistry *CachedRegistry

func NewSchedulerRegistry(interval time.Duration) *CachedRegistry {
	return NewCachedRegistry(fetchSchedulerRegistry, interval)
}

func fetchSchedulerRegistry() (map[string]interface{}, error) {
	// TODO: Implement actual smart contract call to fetch scheduler registry
	// For now, returning a dummy implementation
	return map[string]interface{}{
		"scheduler1": "address1",
		"scheduler2": "address2",
	}, nil
}
