package registry

import (
	"fmt"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/fxnlabs/function-node/internal/config"
	"go.uber.org/zap"
)

// NewSchedulerRegistry creates a new cached registry for schedulers.
func NewSchedulerRegistry(
	client *ethclient.Client,
	cfg *config.Config,
	logger *zap.Logger,
) (*CachedRegistry, error) {
	// For scheduler registry, the ABI might not be complex or even needed if we're just storing addresses.
	// However, to conform to CachedRegistry, we might pass a nil or minimal ABI.
	// Or, if there's a contract call to verify schedulers, an ABI would be needed.
	// For now, assuming a simple case where the "registry" is just the configured address.
	// If actual contract interaction is needed, an ABI and a proper fetch function would be required.

	schedulerContractAddress := common.HexToAddress(cfg.SchedulerAddress)
	specificLogger := logger.Named("scheduler_registry")

	// A dummy ABI if no actual contract calls are made by fetchSchedulerRegistry for now.
	// If fetchSchedulerRegistry needs to call contract methods, provide the actual ABI.
	var dummyABI abi.ABI

	return NewCachedRegistry(
		client,
		schedulerContractAddress,
		dummyABI, // Or actual ABI if needed
		fetchSchedulerRegistry,
		0, // No polling needed for a hardcoded address
		specificLogger,
	), nil
}

// fetchSchedulerRegistry implements FetchFunc.
// For now, it simply returns the configured scheduler address.
// If schedulers are dynamically registered on-chain, this function would query the contract.
func fetchSchedulerRegistry(
	_ *ethclient.Client, // Client might not be used if just returning configured value
	contractAddress common.Address,
	_ abi.ABI, // ABI might not be used for this simple case
	logger *zap.Logger,
) (map[string]interface{}, error) {
	// The "registry" for schedulers, in this simplified version,
	// is just the single configured smart contract address of the scheduler.
	// The key in the map could be the address itself or a predefined key.
	registryMap := make(map[string]interface{})
	if contractAddress == (common.Address{}) {
		logger.Warn("Scheduler smart contract address is not configured or is zero address.")
		// Return an empty map or an error, depending on desired behavior
		return registryMap, fmt.Errorf("scheduler smart contract address is not configured")
	}
	registryMap[contractAddress.Hex()] = contractAddress.Hex() // Storing address as both key and value
	logger.Info("Fetched scheduler registry (configured address)", zap.String("address", contractAddress.Hex()))
	return registryMap, nil
}
