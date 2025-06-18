package registry

import (
	"errors"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/mocks/contracts"
	"github.com/fxnlabs/function-node/mocks/ethclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"go.uber.org/zap"
)

var gatewayABIPath = filepath.Join("../../", GatewayRegistryABIPath)

func TestNewGatewayRegistry(t *testing.T) {
	cfg := &config.Config{}
	cfg.Registry.Gateway.PollInterval = 1 * time.Second
	logger := zap.NewNop()

	t.Run("Success", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockRouter := contracts.NewMockRouter(t)
		mockRouter.EXPECT().GetGatewayRegistryAddress().Return(common.HexToAddress("0x123"), nil)

		// Mock the initial call to fetch the gateway registry
		abiFileContent, err := os.ReadFile(gatewayABIPath)
		assert.NoError(t, err)
		parsedABI, err := abi.JSON(strings.NewReader(string(abiFileContent)))
		assert.NoError(t, err)
		expectedGateways := []Gateway{}
		packedOutput, err := parsedABI.Methods["getActiveGatewaysLive"].Outputs.Pack(expectedGateways)
		assert.NoError(t, err)
		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return(packedOutput, nil)

		registry, err := NewGatewayRegistry(mockClient, cfg, logger, mockRouter, providerABIPath)
		assert.NoError(t, err)
		assert.NotNil(t, registry)
	})

	t.Run("GetGatewayRegistryAddressError", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockRouter := contracts.NewMockRouter(t)
		mockRouter.EXPECT().GetGatewayRegistryAddress().Return(common.Address{}, errors.New("router error"))
		_, err := NewGatewayRegistry(mockClient, cfg, logger, mockRouter, providerABIPath)
		assert.Error(t, err)
	})
}

func TestFetchGatewayRegistry(t *testing.T) {
	logger := zap.NewNop()
	contractAddress := common.HexToAddress("0x123")

	abiFileContent, err := os.ReadFile(gatewayABIPath)
	assert.NoError(t, err)
	parsedABI, err := abi.JSON(strings.NewReader(string(abiFileContent)))
	assert.NoError(t, err)

	t.Run("Success", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		expectedGateways := []Gateway{
			{
				Owner:        common.HexToAddress("0xabc"),
				ID:           []byte("gateway1"),
				RegisteredAt: big.NewInt(12345),
				Metadata:     "meta1",
				Paused:       false,
			},
		}
		packedOutput, err := parsedABI.Methods["getActiveGatewaysLive"].Outputs.Pack(expectedGateways)
		assert.NoError(t, err)

		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return(packedOutput, nil).Once()

		registryMap, err := fetchGatewayRegistry(mockClient, contractAddress, parsedABI, logger)
		assert.NoError(t, err)
		assert.NotNil(t, registryMap)
		assert.Len(t, registryMap, 1)
		assert.Equal(t, expectedGateways[0], registryMap["gateway1"])
	})

	t.Run("CallContractError", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return(nil, errors.New("contract call error")).Once()

		_, err := fetchGatewayRegistry(mockClient, contractAddress, parsedABI, logger)
		assert.Error(t, err)
	})

	t.Run("UnpackError", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		mockClient.EXPECT().CallContract(mock.Anything, mock.Anything, mock.Anything).Return([]byte("invalid data"), nil).Once()

		_, err := fetchGatewayRegistry(mockClient, contractAddress, parsedABI, logger)
		assert.Error(t, err)
	})
}
