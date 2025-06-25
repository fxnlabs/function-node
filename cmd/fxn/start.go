package main

import (
	"fmt"
	"net/http"
	"path/filepath"

	"github.com/ethereum/go-ethereum/common"
	goethclient "github.com/ethereum/go-ethereum/ethclient"
	"github.com/fxnlabs/function-node/internal/auth"
	"github.com/fxnlabs/function-node/internal/challenge"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/contracts"
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/fxnlabs/function-node/internal/metrics"
	"github.com/fxnlabs/function-node/internal/openai"
	"github.com/fxnlabs/function-node/internal/registry"
	"github.com/fxnlabs/function-node/pkg/ethclient"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

func startNode(configHomePath string, cfg *config.Config, ethClient ethclient.EthClient, router contracts.Router, gatewayRegistry registry.Registry, schedulerRegistry registry.Registry, providerRegistry registry.Registry, log *zap.Logger) error {
	rootLogger := log.Named("node")
	modelBackendConfig, err := config.LoadModelBackendConfig(filepath.Join(configHomePath, "model_backend.yaml"))
	if err != nil {
		rootLogger.Fatal("failed to load model_backend config", zap.Error(err))
	}

	privateKey, _, err := keys.LoadPrivateKey(filepath.Join(configHomePath, "nodekey.json"))
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
	http.Handle("/challenge", metrics.Middleware(auth.AuthMiddleware(challengeHandler, rootLogger, nonceCache, schedulerRegistry), "/challenge"))

	oaiProxyHandler := openai.NewOAIProxyHandler(cfg, modelBackendConfig, rootLogger)
	http.Handle("/v1/chat/completions", metrics.Middleware(auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry), "/v1/chat/completions"))
	http.Handle("/v1/completions", metrics.Middleware(auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry), "/v1/completions"))
	http.Handle("/v1/embeddings", metrics.Middleware(auth.AuthMiddleware(oaiProxyHandler, rootLogger, nonceCache, gatewayRegistry), "/v1/embeddings"))

	modelsHandler := openai.NewModelsHandler(modelBackendConfig, rootLogger)
	http.Handle("/v1/models", metrics.Middleware(auth.AuthMiddleware(modelsHandler, rootLogger, nonceCache, gatewayRegistry), "/v1/models"))

	// Expose Prometheus metrics endpoint
	http.Handle("/metrics", promhttp.Handler())
	rootLogger.Info("Prometheus metrics available at /metrics")

	// Health check endpoint
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	addr := fmt.Sprintf(":%d", cfg.Node.ListenPort)
	rootLogger.Info("Starting server on", zap.String("address", addr))
	if err := http.ListenAndServe(addr, nil); err != nil {
		rootLogger.Fatal("failed to start server", zap.Error(err))
	}
	return nil
}

func startCommand() *cli.Command {
	return &cli.Command{
		Name:  "start",
		Usage: "Start the function node",
		Flags: []cli.Flag{
			&cli.BoolFlag{
				Name:  "tui",
				Usage: "Show current identity and log section interactively",
			},
		},
		Action: func(c *cli.Context) error {
			log := c.App.Metadata["logger"].(*zap.Logger)
			cfg := c.App.Metadata["cfg"].(*config.Config)
			homeDir := c.App.Metadata["homeDir"].(string)
			enableTUI := c.Bool("tui")

			// Initialize Ethereum client
			ethClient, err := goethclient.Dial(cfg.RpcProvider)
			if err != nil {
				log.Fatal("Failed to connect to Ethereum RPC provider", zap.String("provider", cfg.RpcProvider), zap.Error(err))
			}
			defer ethClient.Close()

			// Initialize router
			routerAddress := common.HexToAddress(cfg.Registry.RouterSmartContractAddress)
			router, err := contracts.NewRouter(ethClient, routerAddress, log)
			if err != nil {
				log.Fatal("failed to create router", zap.Error(err))
			}

			// Initialize providerRegistry first so it's available for TUI
			providerRegistry, err := registry.NewProviderRegistry(ethClient, cfg, log, router)
			if err != nil {
				log.Fatal("failed to initialize provider registry", zap.Error(err))
			}

			if enableTUI {
				// The TUI reads from Stdin, so it needs to be handled carefully.
				// Running it in a goroutine allows the main app to continue.
				// The TUI itself will handle its lifecycle.
				go StartInteractiveTUI(homeDir, log.Named("tui"), providerRegistry)
				// Add a small delay or a mechanism to ensure TUI initializes before other logs, if necessary.
				// For now, direct launch.
				log.Info("Interactive TUI enabled. Main application logs will continue below/separately.")
			}

			// Initialize other registries
			gatewayRegistry, err := registry.NewGatewayRegistry(ethClient, cfg, log, router)
			if err != nil {
				log.Fatal("failed to initialize gateway registry", zap.Error(err))
			}

			schedulerRegistry, err := registry.NewSchedulerRegistry(ethClient, cfg, log)
			if err != nil {
				log.Fatal("failed to initialize scheduler registry", zap.Error(err))
			}

			// providerRegistry is already initialized above

			return startNode(homeDir, cfg, ethClient, router, gatewayRegistry, schedulerRegistry, providerRegistry, log)
		},
	}
}
