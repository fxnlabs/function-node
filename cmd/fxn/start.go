package main

import (
	"fmt"
	"net/http"

	"github.com/ethereum/go-ethereum/common"
	goethclient "github.com/ethereum/go-ethereum/ethclient"
	"github.com/fxnlabs/function-node/internal/auth"
	"github.com/fxnlabs/function-node/internal/challenge"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/contracts"
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/fxnlabs/function-node/internal/openai"
	"github.com/fxnlabs/function-node/internal/registry"
	"github.com/fxnlabs/function-node/pkg/ethclient"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

func startNode(cfg *config.Config, ethClient ethclient.EthClient, router contracts.Router, gatewayRegistry registry.Registry, schedulerRegistry registry.Registry, providerRegistry registry.Registry, log *zap.Logger) error {
	rootLogger := log.Named("node")
	modelBackendConfig, err := config.LoadModelBackendConfig(cfg.ModelBackendPath)
	if err != nil {
		rootLogger.Fatal("failed to load model_backend config", zap.Error(err))
	}

	privateKey, _, err := keys.LoadPrivateKey(cfg.Node.Keyfile)
	if err != nil {
		rootLogger.Fatal("failed to load private key", zap.Error(err))
	}

	// Use for verifying signatures and preventing replay attacks.
	nonceCache := auth.NewNonceCache(cfg.NonceCache.TTL, cfg.NonceCache.CleanupInterval)

	// Make providerRegistry available if needed by other parts, e.g., auth middleware
	// For now, it's initialized but not explicitly used further in this snippet.
	// If IsProviderRegistered is part of auth, it might use this providerRegistry.
	_ = providerRegistry // Placeholder to use providerRegistry, remove if it's passed to a consumer

	challengeHandler := challenge.ChallengeHandler(rootLogger, privateKey)
	// Assuming AuthMiddleware might need providerRegistry if it performs provider registration checks.
	// If not, schedulerRegistry might be the correct one for challenges.
	// Based on existing code, schedulerRegistry is used for /challenge
	http.Handle("/challenge", auth.AuthMiddleware(challengeHandler, rootLogger, nonceCache, schedulerRegistry))

	oaiProxyHandler := openai.NewOAIProxyHandler(modelBackendConfig, rootLogger)
	http.Handle("/v1/chat/completions", auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry))
	http.Handle("/v1/completions", auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry))
	http.Handle("/v1/embeddings", auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry))

	modelsHandler := openai.NewModelsHandler(modelBackendConfig, rootLogger)
	http.Handle("/v1/models", auth.AuthMiddleware(modelsHandler, rootLogger, nonceCache, gatewayRegistry))

	addr := fmt.Sprintf(":%d", cfg.Node.ListenPort)
	rootLogger.Info("Starting server on", zap.String("address", addr))
	if err := http.ListenAndServe(addr, nil); err != nil {
		rootLogger.Fatal("failed to start server", zap.Error(err))
	}
	return nil
}

func startCommand(log *zap.Logger, cfg *config.Config) *cli.Command {
	return &cli.Command{
		Name:  "start",
		Usage: "Start the function node",
		Action: func(c *cli.Context) error {
			// Initialize Ethereum client
			ethClient, err := goethclient.Dial(cfg.RpcProvider)
			if err != nil {
				log.Fatal("Failed to connect to Ethereum RPC provider", zap.String("provider", cfg.RpcProvider), zap.Error(err))
			}
			defer ethClient.Close()

			// Initialize router
			routerAddress := common.HexToAddress(cfg.Registry.RouterSmartContractAddress)
			router, err := contracts.NewRouter(ethClient, routerAddress, log, contracts.DefaultRouterABIPath)
			if err != nil {
				log.Fatal("failed to create router", zap.Error(err))
			}

			// Initialize registries
			gatewayRegistry, err := registry.NewGatewayRegistry(ethClient, cfg, log, router, registry.GatewayRegistryABIPath)
			if err != nil {
				log.Fatal("failed to initialize gateway registry", zap.Error(err))
			}

			schedulerRegistry, err := registry.NewSchedulerRegistry(ethClient, cfg, log)
			if err != nil {
				log.Fatal("failed to initialize scheduler registry", zap.Error(err))
			}

			providerRegistry, err := registry.NewProviderRegistry(ethClient, cfg, log, router, registry.ProviderRegistryABIPath)
			if err != nil {
				log.Fatal("failed to initialize provider registry", zap.Error(err))
			}

			return startNode(cfg, ethClient, router, gatewayRegistry, schedulerRegistry, providerRegistry, log)
		},
	}
}
