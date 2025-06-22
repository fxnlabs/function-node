package main

import (
	"encoding/json"
	"fmt"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/fxnlabs/function-node/internal/challenge"
	"github.com/fxnlabs/function-node/internal/challenge/challengers"
	"go.uber.org/zap"
)

func main() {
	// Create logger
	logger, _ := zap.NewDevelopment()
	defer logger.Sync()

	// Generate a test private key
	privateKey, err := crypto.GenerateKey()
	if err != nil {
		fmt.Printf("Failed to generate private key: %v\n", err)
		return
	}

	// Test IDENTITY challenge to verify GPU detection
	identityChallenger := challengers.NewIdentityChallenger(privateKey)
	identityPayload := json.RawMessage(`{}`)
	identityReq := challenge.Challenge{
		Type:    "IDENTITY",
		Payload: identityPayload,
	}

	fmt.Println("=== Testing IDENTITY Challenge ===")
	identityResp, err := identityChallenger.Execute(identityReq.Payload, logger)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		respJSON, _ := json.MarshalIndent(identityResp, "", "  ")
		fmt.Printf("Response:\n%s\n\n", string(respJSON))
	}

	// Test MATRIX_MULTIPLICATION challenge
	matrixChallenger := challengers.NewMatrixMultiplicationChallenger()

	// Test 1: Small matrix (256x256)
	fmt.Println("=== Testing MATRIX_MULTIPLICATION Challenge (256x256) ===")
	payload256 := json.RawMessage(`{"size": 256}`)
	req256 := challenge.Challenge{
		Type:    "MATRIX_MULTIPLICATION",
		Payload: payload256,
	}

	resp256, err := matrixChallenger.Execute(req256.Payload, logger)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		respJSON, _ := json.MarshalIndent(resp256, "", "  ")
		fmt.Printf("Response:\n%s\n\n", string(respJSON))
	}

	// Test 2: Medium matrix (1024x1024)
	fmt.Println("=== Testing MATRIX_MULTIPLICATION Challenge (1024x1024) ===")
	payload1024 := json.RawMessage(`{"size": 1024}`)
	req1024 := challenge.Challenge{
		Type:    "MATRIX_MULTIPLICATION",
		Payload: payload1024,
	}

	resp1024, err := matrixChallenger.Execute(req1024.Payload, logger)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		respJSON, _ := json.MarshalIndent(resp1024, "", "  ")
		fmt.Printf("Response:\n%s\n\n", string(respJSON))
	}

	// Test 3: With specific matrices
	fmt.Println("=== Testing MATRIX_MULTIPLICATION Challenge (with specific matrices) ===")
	payloadSpecific := json.RawMessage(`{
		"matrixA": [[1, 2], [3, 4]],
		"matrixB": [[5, 6], [7, 8]]
	}`)
	reqSpecific := challenge.Challenge{
		Type:    "MATRIX_MULTIPLICATION",
		Payload: payloadSpecific,
	}

	respSpecific, err := matrixChallenger.Execute(reqSpecific.Payload, logger)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		respJSON, _ := json.MarshalIndent(respSpecific, "", "  ")
		fmt.Printf("Response:\n%s\n", string(respJSON))
	}
}
