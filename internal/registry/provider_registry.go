package registry

import (
	"context"
	"fmt"
	"math/big"
	"os"
	"strings"

	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/contracts"
	"github.com/fxnlabs/function-node/pkg/ethclient"
	"go.uber.org/zap"
)

const ProviderRegistryABIPath = "fixtures/abi/ProviderRegistry.json"

// Provider represents the structure of a provider in the registry.
type Provider struct {
	Owner        common.Address `json:"owner"`
	Id           []byte         `json:"id"` // Keep as bytes, convert to hex string for map key
	ModelId      *big.Int       `json:"modelId"`
	RegisteredAt *big.Int       `json:"registeredAt"`
	Metadata     string         `json:"metadata"`
	Paused       bool           `json:"paused"`
}

// NewProviderRegistry creates a new cached registry for providers.
func NewProviderRegistry(
	client ethclient.EthClient,
	cfg *config.Config,
	logger *zap.Logger,
	router contracts.Router,
	abiPath string,
) (*CachedRegistry, error) {
	var parsedABI abi.ABI
	var err error
	abiBytes, err := os.ReadFile(abiPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read ProviderRegistry ABI file: %w", err)
	}
	providerRegistryABIJson := string(abiBytes)

	if providerRegistryABIJson == "" {
		return nil, fmt.Errorf("ProviderRegistry ABI JSON is empty")
	}
	parsedABI, err = abi.JSON(strings.NewReader(providerRegistryABIJson))
	if err != nil {
		return nil, fmt.Errorf("failed to parse ProviderRegistry ABI: %w", err)
	}

	providerContractAddress, err := router.GetProviderRegistryAddress()
	if err != nil {
		return nil, fmt.Errorf("failed to get provider registry address: %w", err)
	}
	pollInterval := cfg.Registry.Provider.PollInterval
	specificLogger := logger.Named("provider_registry")

	return NewCachedRegistry(
		client,
		providerContractAddress,
		parsedABI,
		fetchProviderRegistry,
		pollInterval,
		specificLogger,
	), nil
}

// fetchProviderRegistry implements FetchFunc to fetch provider data from the smart contract.
func fetchProviderRegistry(
	client ethclient.EthClient,
	contractAddress common.Address,
	contractABI abi.ABI,
	logger *zap.Logger,
) (map[string]interface{}, error) {
	logger.Info("Fetching provider registry from smart contract", zap.String("contract_address", contractAddress.Hex()))

	// Prepare the call data for "getActiveProvidersLive"
	// This method takes no arguments.
	methodName := "getActiveProvidersLive"
	data, err := contractABI.Pack(methodName)
	if err != nil {
		logger.Error("Failed to pack data for getActiveProvidersLive", zap.Error(err))
		return nil, fmt.Errorf("failed to pack data for %s: %w", methodName, err)
	}

	// Perform the call
	callMsg := ethereum.CallMsg{
		To:   &contractAddress,
		Data: data,
	}
	resultBytes, err := client.CallContract(context.Background(), callMsg, nil)
	if err != nil {
		logger.Error("Failed to call getActiveProvidersLive on contract", zap.Error(err))
		return nil, fmt.Errorf("failed to call %s on contract %s: %w", methodName, contractAddress.Hex(), err)
	}

	// Unpack the result
	// The result is a slice of Provider structs: struct IProviderRegistry.Provider[]
	var providers []Provider
	err = contractABI.UnpackIntoInterface(&providers, methodName, resultBytes)
	if err != nil {
		// Attempt to unpack into a slice of raw structs if direct unpacking fails
		// This can happen if the output is defined as `tuple[]` and contains anonymous structs
		var rawProviders []struct {
			Owner        common.Address `json:"owner"`
			Id           []byte         `json:"id"`
			ModelId      *big.Int       `json:"modelId"`
			RegisteredAt *big.Int       `json:"registeredAt"`
			Metadata     string         `json:"metadata"`
			Paused       bool           `json:"paused"`
		}
		errAlt := contractABI.UnpackIntoInterface(&rawProviders, methodName, resultBytes)
		if errAlt != nil {
			logger.Error("Failed to unpack getActiveProvidersLive result (both direct and raw struct slice)", zap.Error(err), zap.Error(errAlt))
			return nil, fmt.Errorf("failed to unpack %s result (direct: %v, raw: %v)", methodName, err, errAlt)
		}
		// Convert rawProviders to []Provider
		providers = make([]Provider, len(rawProviders))
		for i, rp := range rawProviders {
			providers[i] = Provider(rp)
		}
		logger.Debug("Successfully unpacked getActiveProvidersLive result using raw struct slice")
	} else {
		logger.Debug("Successfully unpacked getActiveProvidersLive result directly")
	}

	// Convert to the required map format
	registryData := make(map[string]interface{})
	for _, provider := range providers {
		registryData[string(provider.Id)] = provider
		logger.Debug("Fetched provider", zap.String("id", string(provider.Id)), zap.String("owner", provider.Owner.Hex()))
	}

	logger.Info("Successfully fetched and processed provider registry", zap.Int("provider_count", len(providers)))
	return registryData, nil
}
