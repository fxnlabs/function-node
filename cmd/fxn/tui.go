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
	"github.com/fxnlabs/function-node/internal/challenge/challengers"
	"github.com/fxnlabs/function-node/internal/keys"
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
	isLogsView        = false
	nodeAddressString string
	publicIPString    string
	gpuInfoString     string
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
	clearStdout()
	myFigure := figure.NewFigure("FxN Node", "", true)
	myFigure.Print()
	fmt.Println("")
	if nodeAddressString != "" {
		fmt.Printf("Node Address: %s\n", nodeAddressString)
	} else {
		fmt.Println("Node Address: Not available")
	}
	if publicIPString != "" {
		fmt.Printf("Public IP: %s\n", publicIPString)
	} else {
		fmt.Println("Public IP: Not available")
	}
	if gpuInfoString != "" {
		fmt.Println("GPU Information:")
		fmt.Print(gpuInfoString) // gpuInfoString includes necessary formatting
	} else {
		fmt.Println("GPU Information: Not available or no GPUs detected.")
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

func fetchAndCacheNodeInfo(homeDir string, baseLog *zap.Logger) {
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
}

// StartInteractiveTUI is intended to be run as a goroutine.
// It handles the display and input for the TUI.
func StartInteractiveTUI(homeDir string, baseLog *zap.Logger) {
	fetchAndCacheNodeInfo(homeDir, baseLog)
	renderIdentityView()

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
