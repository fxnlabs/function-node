package registry

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/fxnlabs/function-node/internal/config"
	"go.uber.org/zap"
)

// Gateway represents the structure of a gateway in the registry
type Gateway struct {
	Owner        common.Address `json:"owner"`
	ID           []byte         `json:"id"` // Note: Solidity bytes type maps to []byte in Go
	RegisteredAt uint64         `json:"registeredAt"`
	Metadata     string         `json:"metadata"`
	Paused       bool           `json:"paused"`
}

// NewGatewayRegistry creates a new cached registry for gateways.
// It now accepts an ethclient.Client, the application config, and a logger.
func NewGatewayRegistry(
	client *ethclient.Client,
	cfg *config.Config,
	logger *zap.Logger,
) (*CachedRegistry, error) {
	// Load ABI content from file
	abiBytes, err := os.ReadFile("fixtures/abi/GatewayRegistry.json")
	if err != nil {
		return nil, fmt.Errorf("failed to read GatewayRegistry ABI file: %w", err)
	}
	gatewayRegistryABIString := string(abiBytes)

	parsedABI, err := abi.JSON(strings.NewReader(gatewayRegistryABIString))
	if err != nil {
		return nil, fmt.Errorf("failed to parse GatewayRegistry ABI: %w", err)
	}

	gatewayContractAddress := common.HexToAddress(cfg.Registry.Gateway.SmartContractAddress)
	pollInterval := cfg.Registry.Gateway.PollInterval
	registryName := "gateway_registry"
	specificLogger := logger.Named(registryName)

	return NewCachedRegistry(
		client,
		gatewayContractAddress,
		parsedABI,
		fetchGatewayRegistry, // Pass the function directly
		pollInterval,
		specificLogger,
	), nil
}

// fetchGatewayRegistry implements the FetchFunc signature.
// It uses the provided client, contractAddress, and contractABI.
func fetchGatewayRegistry(
	client *ethclient.Client,
	contractAddress common.Address,
	contractABI abi.ABI,
	logger *zap.Logger,
) (map[string]interface{}, error) {
	callData, err := contractABI.Pack("getActiveGatewaysLive")
	if err != nil {
		logger.Error("Failed to pack data for getActiveGatewaysLive", zap.Error(err))
		return nil, fmt.Errorf("failed to pack data for getActiveGatewaysLive: %w", err)
	}

	// Use go-ethereum.CallMsg directly
	msg := ethereum.CallMsg{
		To:   &contractAddress,
		Data: callData,
	}
	result, err := client.CallContract(context.Background(), msg, nil)
	if err != nil {
		logger.Error("Failed to call getActiveGatewaysLive", zap.String("contractAddress", contractAddress.Hex()), zap.Error(err))
		return nil, fmt.Errorf("failed to call getActiveGatewaysLive: %w", err)
	}

	var gateways []Gateway
	// The output of getActiveGatewaysLive is an array of Gateway structs.
	// The ABI library should be able to unpack this directly if the Go struct matches the Solidity struct.
	// The method name passed to UnpackIntoInterface should match the method name in the ABI.
	err = contractABI.UnpackIntoInterface(&gateways, "getActiveGatewaysLive", result)
	if err != nil {
		logger.Error("Failed to unpack getActiveGatewaysLive result", zap.Error(err))
		return nil, fmt.Errorf("failed to unpack getActiveGatewaysLive result: %w", err)
	}

	registryMap := make(map[string]interface{})
	for _, gw := range gateways {
		// Assuming the 'id' field of the Gateway struct is a byte slice that can be converted to a string key.
		// You might need to adjust how the key is generated based on the actual structure of 'id'.
		registryMap[string(gw.ID)] = gw
	}

	logger.Info("Successfully fetched gateway registry", zap.Int("count", len(gateways)))
	return registryMap, nil
}
