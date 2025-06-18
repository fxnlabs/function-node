package contracts

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/fxnlabs/function-node/pkg/ethclient"
	"go.uber.org/zap"
)

const (
	DefaultRouterABIPath = "../../fixtures/abi/Router.json"
)

type Router interface {
	GetGatewayRegistryAddress() (common.Address, error)
	GetProviderRegistryAddress() (common.Address, error)
}

type RouterImpl struct {
	client          ethclient.EthClient
	contractAddress common.Address
	contractABI     abi.ABI
	logger          *zap.Logger
}

func NewRouter(client ethclient.EthClient, contractAddress common.Address, logger *zap.Logger, abiPath string) (*RouterImpl, error) {
	abiBytes, err := os.ReadFile(abiPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read Router ABI file: %w", err)
	}
	routerABIString := string(abiBytes)

	parsedABI, err := abi.JSON(strings.NewReader(routerABIString))
	if err != nil {
		return nil, fmt.Errorf("failed to parse Router ABI: %w", err)
	}

	return &RouterImpl{
		client:          client,
		contractAddress: contractAddress,
		contractABI:     parsedABI,
		logger:          logger.Named("router"),
	}, nil
}

func (r *RouterImpl) getAddress(methodName string) (common.Address, error) {
	callData, err := r.contractABI.Pack(methodName)
	if err != nil {
		r.logger.Error("Failed to pack data for get", zap.String("name", methodName), zap.Error(err))
		return common.Address{}, fmt.Errorf("failed to pack data for %s: %w", methodName, err)
	}

	msg := ethereum.CallMsg{
		To:   &r.contractAddress,
		Data: callData,
	}
	result, err := r.client.CallContract(context.Background(), msg, nil)
	if err != nil {
		r.logger.Error("Failed to call get", zap.String("name", methodName), zap.String("contractAddress", r.contractAddress.Hex()), zap.Error(err))
		return common.Address{}, fmt.Errorf("failed to call %s: %w", methodName, err)
	}

	var addr common.Address
	err = r.contractABI.UnpackIntoInterface(&addr, methodName, result)
	if err != nil {
		r.logger.Error("Failed to unpack get result", zap.String("name", methodName), zap.Error(err))
		return common.Address{}, fmt.Errorf("failed to unpack %s result: %w", methodName, err)
	}

	return addr, nil
}

func (r *RouterImpl) GetGatewayRegistryAddress() (common.Address, error) {
	return r.getAddress("gatewayRegistry")
}

func (r *RouterImpl) GetProviderRegistryAddress() (common.Address, error) {
	return r.getAddress("providerRegistry")
}
