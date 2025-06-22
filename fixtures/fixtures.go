package fixtures

import (
	_ "embed"
)

//go:embed abi/GatewayRegistry.json
var GatewayRegistryABI string

//go:embed abi/ProviderRegistry.json
var ProviderRegistryABI string

//go:embed abi/Router.json
var RouterABI string

//go:embed config/config.yaml.template
var ConfigTemplate []byte

//go:embed config/model_backend.yaml.template
var ModelBackendTemplate []byte
