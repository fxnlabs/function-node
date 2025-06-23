package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/common-nighthawk/go-figure"
	"github.com/ethereum/go-ethereum/common"
	"github.com/fxnlabs/function-node/internal/challenge/challengers"
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/fxnlabs/function-node/internal/registry"
	"go.uber.org/zap"
)

// printTuiInteractiveMessage displays a message to the user in the TUI's interactive part.
func printTuiInteractiveMessage(message string) {
	// This ensures interactive messages are distinguishable if needed,
	// and provides a single point for future formatting changes.
	fmt.Printf("[TUI] %s\n", message)
}

var (
	// State for TUI
	isLogsView             = false
	nodeAddressString      string
	publicIPString         string
	gpuInfoString          string
	providerIDString       string
	providerMetadataString string
	providerStatusString   string
)

// clearStdout clears the terminal screen where stdout is directed.
// Note: This is a basic clear and might not behave like sophisticated TUI libraries.
func clearStdout() {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/c", "cls")
	} else {
		cmd = exec.Command("clear")
	}
	cmd.Stdout = os.Stdout
	err := cmd.Run()
	if err != nil {
		// Non-critical, TUI will just print over old content
		// fmt.Fprintf(os.Stderr, "Failed to clear screen: %v\n", err)
	}
}

func renderIdentityView() {
	// Then proceed with clear and full render. This might cause a flicker but shows text fast.
	clearStdout()
	myFigure := figure.NewFigure("FxN Node", "", true)
	myFigure.Print()
	fmt.Println("")
	if nodeAddressString != "" {
		fmt.Printf("Node Address: %s\n", nodeAddressString)
	} else {
		fmt.Println("Node Address: Fetching...")
	}
	if publicIPString != "" {
		fmt.Printf("Public IP: %s\n", publicIPString)
	} else {
		fmt.Println("Public IP: Fetching...")
	}
	if gpuInfoString != "" {
		fmt.Println("GPU Information:")
		fmt.Print(gpuInfoString) // gpuInfoString includes necessary formatting
	} else {
		fmt.Println("GPU Information: Fetching...")
	}
	if providerIDString != "" {
		fmt.Printf("Provider ID: %s\n", providerIDString)
	} else {
		fmt.Println("Provider ID: Not found or fetching...")
	}
	if providerMetadataString != "" {
		fmt.Printf("Provider Metadata: %s\n", providerMetadataString)
	} else {
		fmt.Println("Provider Metadata: Not found or fetching...")
	}
	if providerStatusString != "" {
		fmt.Printf("Provider Status: %s\n", providerStatusString)
	} else {
		fmt.Println("Provider Status: Not found or fetching...")
	}
	fmt.Println("-----------------------------------------------")
	fmt.Println("Press 'l' + Enter to toggle logs view. App logs stream separately (usually stderr).")
}

func renderLogsHeaderView() {
	clearStdout()
	myFigure := figure.NewFigure("FxN Logs", "", true)
	myFigure.Print()
	fmt.Println("")
	fmt.Println("Application logs are streaming (usually to stderr).")
	fmt.Println("Press 'l' + Enter to toggle identity view.")
	fmt.Println("-----------------------------------------------")
}

func fetchAndCacheNodeInfo(homeDir string, baseLog *zap.Logger, providerRegistry registry.Registry) {
	logger := baseLog.Named("tui-fetch")
	keyPath := filepath.Join(homeDir, "nodekey.json")

	// Attempt to load private key and derive address first
	privateKey, nodeAddrFromKey, errPk := keys.LoadPrivateKey(keyPath)
	if errPk == nil {
		nodeAddressString = nodeAddrFromKey.Hex() // Tentatively set from key
	} else {
		logger.Warn("TUI: Failed to load private key. Node address from key won't be available.", zap.Error(errPk))
		// privateKey will be nil, nodeAddressString remains empty
	}

	identityChallenger := challengers.NewIdentityChallenger(privateKey) // privateKey can be nil
	identityInfo, errExec := identityChallenger.Execute(nil, logger)    // Logs from Execute go to stderr via logger

	if errExec == nil { // Execute succeeded
		if infoMap, ok := identityInfo.(map[string]interface{}); ok {
			// Try to get publicKey from challenger; this is preferred
			if pubKey, ok := infoMap["publicKey"].(string); ok && pubKey != "" {
				nodeAddressString = pubKey // Override if challenger provides it
			} else if nodeAddressString == "" {
				// Challenger succeeded but didn't provide publicKey, and key load also failed or address was empty.
				logger.Warn("TUI: Challenger executed but no publicKey found, and key loading also failed or address was empty.")
			}
			// else: challenger succeeded, no publicKey, but we might have nodeAddressString from the key file.

			// Set other info that comes from the challenger
			if ip, ok := infoMap["ipAddress"].(string); ok {
				publicIPString = ip
			}
			if gpuStats, ok := infoMap["gpuStats"].([]challengers.GPUStat); ok && len(gpuStats) > 0 {
				var sb strings.Builder
				for i, gpu := range gpuStats {
					sb.WriteString(fmt.Sprintf("   GPU %d: %s\n", i, gpu.Name))
					if gpu.DriverVersion != "" && gpu.DriverVersion != "N/A" {
						sb.WriteString(fmt.Sprintf("     Driver: %s\n", gpu.DriverVersion))
					}
					if gpu.VRAMTotalMB > 0 {
						sb.WriteString(fmt.Sprintf("     VRAM: %d MB Total", gpu.VRAMTotalMB))
						if gpu.UnifiedMemory {
							sb.WriteString(" (Unified)")
						}
						sb.WriteString("\n")
					}
				}
				gpuInfoString = sb.String()
			}
		} else {
			logger.Warn("TUI: Challenger executed but identityInfo is not of expected type map[string]interface{}.")
			// nodeAddressString would retain value from key load, if any. publicIP and GPU info remain empty.
		}
	} else { // Execute failed
		logger.Warn("TUI: Failed to retrieve full identity information from challenger.", zap.Error(errExec))
		// nodeAddressString retains value from key load, if any.
		// publicIPString and gpuInfoString will remain empty as they depend on successful challenger execution.
	}

	// Fetch provider info from registry
	if providerRegistry != nil && nodeAddressString != "" {
		logger.Info("Attempting to fetch provider info from registry using Get()", zap.String("nodeAddress", nodeAddressString))
		nodeAddrLower := strings.ToLower(nodeAddressString) // Ensure consistent casing for lookup
		p, found := providerRegistry.Get(nodeAddrLower)

		if found {
			if provider, ok := p.(registry.Provider); ok {
				// Ensure the Owner address from the retrieved provider matches, just in case of case sensitivity in map keys
				// (though Hex() should be consistent, this is an extra check)
				if strings.ToLower(provider.Owner.Hex()) == nodeAddrLower {
					providerIDString = common.Bytes2Hex(provider.Id) // Convert ID bytes to hex string
					providerMetadataString = provider.Metadata
					if provider.Paused {
						providerStatusString = "Paused"
					} else {
						providerStatusString = "Active"
					}
					logger.Info("Found matching provider in registry",
						zap.String("providerID", providerIDString),
						zap.String("owner", provider.Owner.Hex()),
						zap.String("metadata", providerMetadataString),
						zap.Bool("paused", provider.Paused))
				} else {
					// This case should ideally not happen if keys are stored consistently.
					logger.Warn("Provider found by Get(), but owner address mismatch after case normalization",
						zap.String("requestedAddress", nodeAddrLower),
						zap.String("providerOwner", strings.ToLower(provider.Owner.Hex())))
				}
			} else {
				logger.Warn("Item retrieved from provider registry is not of type registry.Provider", zap.Any("item", p))
			}
		} else {
			logger.Info("No provider found in registry for node address using Get()", zap.String("nodeAddress", nodeAddrLower))
			// providerIDString, providerMetadataString, providerStatusString will remain empty
		}
	} else {
		if providerRegistry == nil {
			logger.Warn("Provider registry is nil, cannot fetch provider info.")
		}
		if nodeAddressString == "" {
			logger.Warn("Node address is empty, cannot fetch provider info.")
		}
	}
}

// StartInteractiveTUI is intended to be run as a goroutine.
// It handles the display and input for the TUI.
func StartInteractiveTUI(homeDir string, baseLog *zap.Logger, providerRegistry registry.Registry) {
	renderIdentityView() // Render initial view immediately

	// Fetch node info in the background
	go func() {
		fetchAndCacheNodeInfo(homeDir, baseLog, providerRegistry)
		// After fetching, re-render if still in identity view
		if !isLogsView {
			// Need to ensure prompt is not overwritten if user is typing
			// A simple way is to clear and re-render.
			// For a more sophisticated TUI, a channel/event system would be better.
			fmt.Println() // Add a newline to avoid clobbering potential user input line
			renderIdentityView()
			fmt.Print("> ") // Re-print prompt
		}
	}()

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ") // Prompt for TUI command
		input, err := reader.ReadString('\n')
		if err != nil {
			baseLog.Error("TUI input error, exiting TUI loop.", zap.Error(err))
			return // Exit TUI goroutine
		}

		input = strings.TrimSpace(strings.ToLower(input))
		if input == "l" {
			isLogsView = !isLogsView
			if isLogsView {
				renderLogsHeaderView()
			} else {
				renderIdentityView()
			}
		} else if input != "" {
			// Handle other commands or provide help
			printTuiInteractiveMessage("Unknown command. Press 'l' + Enter to toggle views.")
		}
		// If input is empty (just Enter), re-prompt without changing view.
	}
}
