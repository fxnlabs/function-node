package registry

import (
	"testing"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/mocks/ethclient"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func TestNewSchedulerRegistry(t *testing.T) {
	logger := zap.NewNop()

	t.Run("Success", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		cfg := &config.Config{
			SchedulerAddress: "0x123",
		}
		registry, err := NewSchedulerRegistry(mockClient, cfg, logger)
		assert.NoError(t, err)
		assert.NotNil(t, registry)
	})
}

func TestFetchSchedulerRegistry(t *testing.T) {
	logger := zap.NewNop()

	t.Run("Success", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		contractAddress := common.HexToAddress("0x123")

		registryMap, err := fetchSchedulerRegistry(mockClient, contractAddress, abi.ABI{}, logger)
		assert.NoError(t, err)
		assert.NotNil(t, registryMap)
		assert.Len(t, registryMap, 1)
		assert.Equal(t, contractAddress.Hex(), registryMap[contractAddress.Hex()])
	})

	t.Run("EmptyAddress", func(t *testing.T) {
		mockClient := ethclient.NewMockEthClient(t)
		contractAddress := common.Address{}

		_, err := fetchSchedulerRegistry(mockClient, contractAddress, abi.ABI{}, logger)
		assert.EqualError(t, err, "scheduler smart contract address is not configured")
	})
}
