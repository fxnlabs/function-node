package main

import (
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

func accountCommands(log *zap.Logger) *cli.Command {
	return &cli.Command{
		Name:  "account",
		Usage: "Manage account",
		Subcommands: []*cli.Command{
			{
				Name:  "new",
				Usage: "Create a new account",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:  "out",
						Value: "nodekey.json",
						Usage: "path to save the key file",
					},
				},
				Action: func(c *cli.Context) error {
					return keys.GenerateKeyFile(c.String("out"))
				},
			},
			{
				Name:  "get",
				Usage: "Get the account address",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:  "in",
						Value: "nodekey.json",
						Usage: "path to the key file",
					},
				},
				Action: func(c *cli.Context) error {
					_, address, err := keys.LoadPrivateKey(c.String("in"))
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
