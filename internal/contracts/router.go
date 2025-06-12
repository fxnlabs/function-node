package contracts

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"go.uber.org/zap"
)

type Router struct {
	client          *ethclient.Client
	contractAddress common.Address
	contractABI     abi.ABI
	logger          *zap.Logger
}

func NewRouter(client *ethclient.Client, contractAddress common.Address, logger *zap.Logger) (*Router, error) {
	abiBytes, err := os.ReadFile("fixtures/abi/Router.json")
	if err != nil {
		return nil, fmt.Errorf("failed to read Router ABI file: %w", err)
	}
	routerABIString := string(abiBytes)

	parsedABI, err := abi.JSON(strings.NewReader(routerABIString))
	if err != nil {
		return nil, fmt.Errorf("failed to parse Router ABI: %w", err)
	}

	return &Router{
		client:          client,
		contractAddress: contractAddress,
		contractABI:     parsedABI,
		logger:          logger.Named("router"),
	}, nil
}

func (r *Router) getAddress(name string) (common.Address, error) {
	callData, err := r.contractABI.Pack("get", name)
	if err != nil {
		r.logger.Error("Failed to pack data for get", zap.String("name", name), zap.Error(err))
		return common.Address{}, fmt.Errorf("failed to pack data for get: %w", err)
	}

	msg := ethereum.CallMsg{
		To:   &r.contractAddress,
		Data: callData,
	}
	result, err := r.client.CallContract(context.Background(), msg, nil)
	if err != nil {
		r.logger.Error("Failed to call get", zap.String("name", name), zap.String("contractAddress", r.contractAddress.Hex()), zap.Error(err))
		return common.Address{}, fmt.Errorf("failed to call get: %w", err)
	}

	var addr common.Address
	err = r.contractABI.UnpackIntoInterface(&addr, "get", result)
	if err != nil {
		r.logger.Error("Failed to unpack get result", zap.String("name", name), zap.Error(err))
		return common.Address{}, fmt.Errorf("failed to unpack get result: %w", err)
	}

	return addr, nil
}

func (r *Router) GetGatewayRegistryAddress() (common.Address, error) {
	return r.getAddress("GatewayRegistry")
}

func (r *Router) GetProviderRegistryAddress() (common.Address, error) {
	return r.getAddress("ProviderRegistry")
}

func (r *Router) GetSchedulerRegistryAddress() (common.Address, error) {
	return r.getAddress("SchedulerRegistry")
}
