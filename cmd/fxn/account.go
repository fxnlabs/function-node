package main

import (
	"path/filepath"

	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

const keyFileName = "nodekey.json"

func accountCommands() *cli.Command {
	return &cli.Command{
		Name:  "account",
		Usage: "Manage account",
		Subcommands: []*cli.Command{
			{
				Name:  "new",
				Usage: "Create a new account",
				Action: func(c *cli.Context) error {
					homeDir := c.App.Metadata["homeDir"].(string)
					return keys.GenerateKeyFile(filepath.Join(homeDir, keyFileName))
				},
			},
			{
				Name:  "get",
				Usage: "Get the account address",
				Action: func(c *cli.Context) error {
					log := c.App.Metadata["logger"].(*zap.Logger)
					homeDir := c.App.Metadata["homeDir"].(string)
					_, address, err := keys.LoadPrivateKey(filepath.Join(homeDir, keyFileName))
					if err != nil {
						return err
					}
					log.Info("Account address", zap.String("address", address.Hex()))
					return nil
				},
			},
		},
	}
}
