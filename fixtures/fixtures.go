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