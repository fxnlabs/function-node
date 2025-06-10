package main

import (
	"os"

	"github.com/ethereum/go-ethereum/accounts/keystore"
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
				Name:  "accounts",
				Usage: "Manage accounts",
				Subcommands: []*cli.Command{
					{
						Name:  "new",
						Usage: "Create a new account",
						Action: func(c *cli.Context) error {
							ks := keystore.NewKeyStore("./keystore", keystore.StandardScryptN, keystore.StandardScryptP)
							password := "password" // This should be prompted from the user
							account, err := ks.NewAccount(password)
							if err != nil {
								return err
							}
							log.Info("New account created", zap.String("address", account.Address.Hex()))
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
