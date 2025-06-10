package registry

import (
	// Keep for potential future use with ABI parsing errors
	// "os" // Uncomment if an ABI file is used for provider registry
	// "strings" // Uncomment if an ABI file is used for provider registry

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/fxnlabs/function-node/internal/config"
	"go.uber.org/zap"
)

// Provider represents the structure of a provider in the registry (if needed for actual contract calls)
// type Provider struct {
//    Owner common.Address `json:"owner"`
//    ID    string         `json:"id"`
//    ...
// }

// TODO: Define providerRegistryABIJson if an ABI file is used.
// var providerRegistryABIJson string

// func init() {
//  abiBytes, err := os.ReadFile("fixtures/abi/ProviderRegistry.json") // Adjust path
//  if err != nil {
//      panic(fmt.Errorf("failed to read ProviderRegistry ABI file: %w", err))
//  }
//  providerRegistryABIJson = string(abiBytes)
// }

// NewProviderRegistry creates a new cached registry for providers.
func NewProviderRegistry(
	client *ethclient.Client,
	cfg *config.Config,
	logger *zap.Logger,
) (*CachedRegistry, error) {
	var parsedABI abi.ABI
	// var err error // err is declared but not used if ABI parsing is commented out

	// if providerRegistryABIJson != "" { // Uncomment if ABI is used
	// 	var parseErr error
	// 	parsedABI, parseErr = abi.JSON(strings.NewReader(providerRegistryABIJson))
	// 	if parseErr != nil {
	// 		return nil, fmt.Errorf("failed to parse ProviderRegistry ABI: %w", parseErr)
	// 	}
	// }

	providerContractAddress := common.HexToAddress(cfg.Registry.Provider.SmartContractAddress)
	pollInterval := cfg.Registry.Provider.PollInterval
	registryName := "provider_registry"
	specificLogger := logger.Named(registryName)

	return NewCachedRegistry(
		client,
		providerContractAddress,
		parsedABI, // Pass parsedABI, which could be zero if no ABI string is defined/loaded
		fetchProviderRegistry,
		pollInterval,
		specificLogger,
	), nil
}

// fetchProviderRegistry implements FetchFunc.
// TODO: Implement actual smart contract call to fetch provider registry.
func fetchProviderRegistry(
	_ *ethclient.Client, // client will be used when actual contract calls are made
	contractAddress common.Address,
	_ abi.ABI, // contractABI will be used when actual contract calls are made
	logger *zap.Logger,
) (map[string]interface{}, error) {
	logger.Info("Fetching provider registry (dummy implementation)", zap.String("contract_address", contractAddress.Hex()))
	// For now, returning a dummy implementation
	// Replace this with actual smart contract interaction logic
	return map[string]interface{}{
		"provider1_dummy_id": "0xProvider1DummyAddress",
		"provider2_dummy_id": "0xProvider2DummyAddress",
	}, nil
}
