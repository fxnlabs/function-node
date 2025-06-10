package registry

import (
	"sync"
	"time"
)

// CachedRegistry provides a generic caching mechanism for registry data.
type CachedRegistry struct {
	mu         sync.RWMutex
	cache      map[string]interface{}
	updateFunc func() (map[string]interface{}, error)
	interval   time.Duration
}

// NewCachedRegistry creates and starts a new cached registry.
func NewCachedRegistry(updateFunc func() (map[string]interface{}, error), interval time.Duration) *CachedRegistry {
	cr := &CachedRegistry{
		cache:      make(map[string]interface{}),
		updateFunc: updateFunc,
		interval:   interval,
	}
	go cr.run()
	return cr
}

// run periodically updates the cache.
func (cr *CachedRegistry) run() {
	ticker := time.NewTicker(cr.interval)
	defer ticker.Stop()

	for {
		cr.updateCache()
		<-ticker.C
	}
}

// updateCache fetches the latest data and updates the cache.
func (cr *CachedRegistry) updateCache() {
	// TODO: Implement actual smart contract call
	data, err := cr.updateFunc()
	if err != nil {
		// TODO: Add logging
		return
	}

	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.cache = data
}

// Get returns a value from the cache.
func (cr *CachedRegistry) Get(key string) (interface{}, bool) {
	cr.mu.RLock()
	defer cr.mu.RUnlock()
	val, ok := cr.cache[key]
	return val, ok
}
