package registry

import (
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"go.uber.org/zap"
)

// FetchFunc defines the signature for functions that fetch registry data.
type FetchFunc func(client *ethclient.Client, contractAddress common.Address, contractABI abi.ABI, logger *zap.Logger) (map[string]interface{}, error)

// CachedRegistry provides a generic caching mechanism for registry data.
type CachedRegistry struct {
	mu              sync.RWMutex
	cache           map[string]interface{}
	client          *ethclient.Client
	contractAddress common.Address
	contractABI     abi.ABI
	fetchFunc       FetchFunc
	interval        time.Duration
	logger          *zap.Logger
}

// NewCachedRegistry creates and starts a new cached registry.
func NewCachedRegistry(
	client *ethclient.Client,
	contractAddress common.Address,
	contractABI abi.ABI,
	fetchFunc FetchFunc,
	interval time.Duration,
	logger *zap.Logger,
) *CachedRegistry {
	cr := &CachedRegistry{
		cache:           make(map[string]interface{}),
		client:          client,
		contractAddress: contractAddress,
		contractABI:     contractABI,
		fetchFunc:       fetchFunc,
		interval:        interval,
		logger:          logger.Named("cached_registry"),
	}
	// Initial fetch to populate cache immediately
	cr.updateCache()
	go cr.run()
	return cr
}

// run periodically updates the cache.
func (cr *CachedRegistry) run() {
	if cr.interval == 0 {
		cr.logger.Info("Cache polling interval is zero, cache will not be updated periodically.")
		return // Do not start the ticker if interval is zero
	}
	ticker := time.NewTicker(cr.interval)
	defer ticker.Stop()

	for {
		<-ticker.C
		cr.updateCache()
	}
}

// updateCache fetches the latest data and updates the cache.
func (cr *CachedRegistry) updateCache() {
	data, err := cr.fetchFunc(cr.client, cr.contractAddress, cr.contractABI, cr.logger)
	if err != nil {
		cr.logger.Error("Failed to update cache",
			zap.String("contract_address", cr.contractAddress.Hex()),
			zap.Error(err))
		return
	}

	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.cache = data
	cr.logger.Info("Cache updated successfully", zap.String("contract_address", cr.contractAddress.Hex()), zap.Int("item_count", len(data)))
}

// Get returns a value from the cache.
func (cr *CachedRegistry) Get(key string) (interface{}, bool) {
	cr.mu.RLock()
	defer cr.mu.RUnlock()
	val, ok := cr.cache[key]
	return val, ok
}
