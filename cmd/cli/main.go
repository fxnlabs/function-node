package main

import (
	"crypto/ecdsa"
	"encoding/hex"
	"os"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

func main() {
	var log *zap.Logger
	app := &cli.App{
		Name:  "fxn",
		Usage: "A CLI for interacting with the Function Network",
		Before: func(c *cli.Context) error {
			cfg, err := config.LoadConfig(c.String("config"))
			if err != nil {
				return err
			}
			zapLogger, err := logger.New(cfg.Logger.Verbosity)
			if err != nil {
				return err
			}
			log = zapLogger.Named("cli")
			return nil
		},
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "config",
				Value:   "config.yaml",
				Usage:   "Load configuration from `FILE`",
				EnvVars: []string{"FXN_CONFIG"},
			},
		},
		Commands: []*cli.Command{
			{
				Name:  "account",
				Usage: "Manage account",
				Subcommands: []*cli.Command{
					{
						Name:  "new",
						Usage: "Create a new account",
						Action: func(c *cli.Context) error {
							privateKey, err := crypto.GenerateKey()
							if err != nil {
								return err
							}

							privateKeyBytes := crypto.FromECDSA(privateKey)
							err = os.WriteFile("nodekey", []byte(hex.EncodeToString(privateKeyBytes)), 0600)
							if err != nil {
								return err
							}

							publicKey := privateKey.Public()
							publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
							if !ok {
								log.Fatal("error casting public key to ECDSA")
							}

							log.Info("New account created", zap.String("address", crypto.PubkeyToAddress(*publicKeyECDSA).Hex()))
							return nil
						},
					},
					{
						Name:  "get",
						Usage: "Get the account address",
						Action: func(c *cli.Context) error {
							privateKeyBytes, err := os.ReadFile("nodekey")
							if err != nil {
								return err
							}
							privateKey, err := crypto.HexToECDSA(string(privateKeyBytes))
							if err != nil {
								return err
							}
							publicKey := privateKey.Public()
							publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
							if !ok {
								log.Fatal("error casting public key to ECDSA")
							}
							log.Info("Account address", zap.String("address", crypto.PubkeyToAddress(*publicKeyECDSA).Hex()))
							return nil
						},
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		if log != nil {
			log.Fatal("failed to run app", zap.Error(err))
		} else {
			panic(err)
		}
	}
}
