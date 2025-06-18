package challengers

import (
	"crypto/ecdsa"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/ethereum/go-ethereum/crypto"
	"go.uber.org/zap"
)

// Executor interface for running external commands
type Executor interface {
	Command(name string, arg ...string) *exec.Cmd
}

type GPUStat struct {
	Name              string  `json:"name"`
	DriverVersion     string  `json:"driver_version"`
	VRAMTotalMB       int     `json:"vram_total_mb"`
	VRAMUsedMB        int     `json:"vram_used_mb"`
	VRAMFreeMB        int     `json:"vram_free_mb"`
	UtilizationGPUPct float64 `json:"utilization_gpu_pct"`
	UtilizationMemPct float64 `json:"utilization_mem_pct"`
	UnifiedMemory     bool    `json:"unified_memory,omitempty"`
}

// IdentityChallenger provides a challenge that returns the node's identity.
type IdentityChallenger struct {
	privateKey *ecdsa.PrivateKey
	Client     *http.Client
	exec       Executor
	goos       string
}

type CMDExecutor struct{}

func (e *CMDExecutor) Command(name string, arg ...string) *exec.Cmd {
	return exec.Command(name, arg...)
}

// NewIdentityChallenger creates a new IdentityChallenger.
func NewIdentityChallenger(privateKey *ecdsa.PrivateKey) *IdentityChallenger {
	return &IdentityChallenger{
		privateKey: privateKey,
		Client:     http.DefaultClient,
		exec:       &CMDExecutor{},
		goos:       runtime.GOOS,
	}
}

// Execute returns the public key, IP address, and a signature.
func (c *IdentityChallenger) Execute(payload interface{}, log *zap.Logger) (interface{}, error) {
	resp, err := c.Client.Get("https://api.ipify.org")
	if err != nil {
		log.Error("Failed to get public IP address", zap.Error(err))
		return nil, err
	}
	defer resp.Body.Close()

	ip, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("Failed to read public IP address response", zap.Error(err))
		return nil, err
	}

	gpuStats, err := c.getGPUStats(log)
	if err != nil {
		// Not all nodes will have GPUs, so we log the error but don't fail the challenge
		log.Warn("Could not get GPU stats", zap.Error(err))
	}

	publicKey := c.privateKey.Public()
	publicKeyBytes := crypto.FromECDSAPub(publicKey.(*ecdsa.PublicKey))

	message := []byte(fmt.Sprintf("%s.%s", publicKeyBytes, ip))
	hash := crypto.Keccak256(message)
	signature, err := crypto.Sign(hash, c.privateKey)
	if err != nil {
		log.Error("Failed to sign identity message", zap.Error(err))
		return nil, err
	}

	return map[string]interface{}{
		"publicKey": crypto.PubkeyToAddress(*publicKey.(*ecdsa.PublicKey)).Hex(),
		"ipAddress": string(ip),
		"signature": signature,
		"gpuStats":  gpuStats,
	}, nil
}

func (c *IdentityChallenger) getGPUStats(log *zap.Logger) ([]GPUStat, error) {
	log.Info("Polling GPU stats...")

	var output []byte
	var err error

	var gpuType string
	switch c.goos {
	case "darwin":
		output, err = c.getMacGPUStats(log)
		gpuType = "mac"
	case "linux":
		// For Linux, try nvidia-smi first, then rocm-smi
		output, err = c.getNvidiaGPUStats(log)
		gpuType = "nvidia"
		if err != nil {
			log.Warn("nvidia-smi failed, trying rocm-smi", zap.Error(err))
			output, err = c.getAmdGPUStats(log)
			gpuType = "amd"
		}
	default:
		// Fallback to nvidia-smi for other OSes
		output, err = c.getNvidiaGPUStats(log)
		gpuType = "nvidia"
	}

	if err != nil {
		log.Error("Failed to get GPU stats", zap.Error(err))
		return nil, err
	}

	gpuStats, err := parseGPUStats(output, gpuType, log, c)
	if err != nil {
		log.Error("Failed to parse GPU stats", zap.Error(err))
		return nil, err
	}

	log.Info("Successfully polled GPU stats")
	return gpuStats, nil
}

func (c *IdentityChallenger) getMacGPUStats(log *zap.Logger) ([]byte, error) {
	cmd := c.exec.Command("system_profiler", "SPDisplaysDataType")
	output, err := cmd.Output()
	if err != nil {
		log.Error("system_profiler failed", zap.Error(err))
		return nil, err
	}
	return output, nil
}

func (c *IdentityChallenger) getNvidiaGPUStats(log *zap.Logger) ([]byte, error) {
	cmd := c.exec.Command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			log.Error("nvidia-smi failed", zap.Error(exitErr), zap.String("stderr", string(exitErr.Stderr)))
			// nvidia-smi not found, handle this case gracefully
			if strings.Contains(string(exitErr.Stderr), "executable file not found") || strings.Contains(string(exitErr.Stderr), "No such file or directory") {
				log.Warn("nvidia-smi command not found, skipping GPU stats poll")
				return nil, fmt.Errorf("nvidia-smi not found")
			}
			return nil, err
		}
		log.Error("Failed to execute nvidia-smi", zap.Error(err))
		return nil, err
	}
	return output, nil
}

func (c *IdentityChallenger) getAmdGPUStats(log *zap.Logger) ([]byte, error) {
	cmd := c.exec.Command("rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv")
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			log.Error("rocm-smi failed", zap.Error(exitErr), zap.String("stderr", string(exitErr.Stderr)))
			if strings.Contains(string(exitErr.Stderr), "executable file not found") || strings.Contains(string(exitErr.Stderr), "No such file or directory") {
				log.Warn("rocm-smi command not found, skipping GPU stats poll")
				return nil, fmt.Errorf("rocm-smi not found")
			}
			return nil, err
		}
		log.Error("Failed to execute rocm-smi", zap.Error(err))
		return nil, err
	}
	return output, nil
}

func (c *IdentityChallenger) getMacSystemMemory(log *zap.Logger) (uint64, error) {
	cmd := c.exec.Command("sysctl", "-n", "hw.memsize")
	output, err := cmd.Output()
	if err != nil {
		log.Error("sysctl hw.memsize failed", zap.Error(err))
		return 0, err
	}
	mem, err := strconv.ParseUint(strings.TrimSpace(string(output)), 10, 64)
	if err != nil {
		log.Error("failed to parse hw.memsize", zap.Error(err))
		return 0, err
	}
	return mem, nil
}

func (c *IdentityChallenger) getMacMemoryUsage(log *zap.Logger) (usedMB int, freeMB int, err error) {
	cmd := c.exec.Command("vm_stat")
	output, err := cmd.Output()
	if err != nil {
		log.Error("failed to run vm_stat", zap.Error(err))
		return 0, 0, err
	}

	lines := strings.Split(string(output), "\n")
	var pageSize int
	var freePages, activePages, inactivePages, wiredPages int

	rePageSize := regexp.MustCompile(`page size of (\d+) bytes`)
	if matches := rePageSize.FindStringSubmatch(lines[0]); len(matches) > 1 {
		pageSize, _ = strconv.Atoi(matches[1])
	} else {
		return 0, 0, fmt.Errorf("could not parse page size from vm_stat")
	}

	for _, line := range lines {
		parts := strings.Split(line, ":")
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			valStr := strings.Trim(strings.TrimSpace(parts[1]), ".")
			val, err := strconv.Atoi(valStr)
			if err != nil {
				log.Warn("could not parse vm_stat value", zap.String("value", valStr))
				continue
			}

			switch key {
			case "Pages free":
				freePages = val
			case "Pages active":
				activePages = val
			case "Pages inactive":
				inactivePages = val
			case "Pages wired down":
				wiredPages = val
			}
		}
	}

	usedPages := activePages + inactivePages + wiredPages
	usedBytes := usedPages * pageSize
	freeBytes := freePages * pageSize

	return usedBytes / 1024 / 1024, freeBytes / 1024 / 1024, nil
}

func parseGPUStats(output []byte, gpuType string, log *zap.Logger, c *IdentityChallenger) ([]GPUStat, error) {
	switch gpuType {
	case "mac":
		return c.parseMacGPUStats(output, log)
	case "nvidia":
		return parseNvidiaGPUStats(output, log)
	case "amd":
		return parseAmdGPUStats(output, log)
	default:
		return nil, fmt.Errorf("unsupported gpu type: %s", gpuType)
	}
}

func (c *IdentityChallenger) parseMacGPUStats(output []byte, log *zap.Logger) ([]GPUStat, error) {
	var stats []GPUStat
	lines := strings.Split(string(output), "\n")
	var currentGPU *GPUStat
	isAppleSilicon := false

	cmd := c.exec.Command("sw_vers", "-productVersion")
	ver, err := cmd.Output()
	driverVersion := "N/A"
	if err == nil {
		driverVersion = strings.TrimSpace(string(ver))
	}

	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "Chipset Model:") {
			if currentGPU != nil {
				stats = append(stats, *currentGPU)
			}
			model := strings.TrimSpace(strings.Split(trimmedLine, ":")[1])
			currentGPU = &GPUStat{
				Name:          model,
				DriverVersion: driverVersion,
			}
			if strings.Contains(model, "Apple M") {
				isAppleSilicon = true
				currentGPU.UnifiedMemory = true
			}
		} else if strings.HasPrefix(trimmedLine, "VRAM (Total):") && currentGPU != nil {
			memStr := strings.TrimSpace(strings.Split(trimmedLine, ":")[1])
			vramMb, err := parseMemoryString(memStr)
			if err != nil {
				return nil, err
			}
			currentGPU.VRAMTotalMB = vramMb
		}
	}

	if currentGPU != nil {
		stats = append(stats, *currentGPU)
	}

	if len(stats) > 0 && isAppleSilicon {
		memBytes, err := c.getMacSystemMemory(log)
		if err == nil {
			memMB := memBytes / 1024 / 1024
			stats[0].VRAMTotalMB = int(memMB)
		}

		usedMB, _, err := c.getMacMemoryUsage(log)
		if err == nil {
			stats[0].VRAMUsedMB = usedMB
			if stats[0].VRAMTotalMB > 0 {
				stats[0].VRAMFreeMB = stats[0].VRAMTotalMB - usedMB
				utilization := (float64(usedMB) / float64(stats[0].VRAMTotalMB)) * 100
				stats[0].UtilizationGPUPct = float64(int(utilization*100)) / 100
				stats[0].UtilizationMemPct = stats[0].UtilizationGPUPct
			}
		}
	}

	return stats, nil
}

func parseNvidiaGPUStats(output []byte, log *zap.Logger) ([]GPUStat, error) {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	var stats []GPUStat
	for _, line := range lines {
		values := strings.Split(line, ", ")
		if len(values) < 7 {
			log.Warn("Unexpected nvidia-smi format", zap.String("line", line))
			continue
		}
		total, _ := strconv.Atoi(strings.TrimSpace(values[2]))
		used, _ := strconv.Atoi(strings.TrimSpace(values[3]))
		free, _ := strconv.Atoi(strings.TrimSpace(values[4]))
		gpuPct, _ := strconv.ParseFloat(strings.TrimSpace(values[5]), 64)
		memPct, _ := strconv.ParseFloat(strings.TrimSpace(values[6]), 64)
		gpuStat := GPUStat{
			Name:              values[0],
			DriverVersion:     values[1],
			VRAMTotalMB:       total,
			VRAMUsedMB:        used,
			VRAMFreeMB:        free,
			UtilizationGPUPct: gpuPct,
			UtilizationMemPct: memPct,
		}
		stats = append(stats, gpuStat)
	}
	return stats, nil
}

func parseAmdGPUStats(output []byte, log *zap.Logger) ([]GPUStat, error) {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 0 || (len(lines) == 1 && lines[0] == "") {
		return []GPUStat{}, nil
	}
	var stats []GPUStat

	startLine := 0
	// Heuristically check for a header line. If the memory field isn't a number, it's probably a header.
	firstLineValues := strings.Split(lines[0], ",")
	if len(firstLineValues) >= 4 {
		if _, err := strconv.Atoi(strings.TrimSpace(firstLineValues[2])); err != nil {
			startLine = 1
		}
	}

	for _, line := range lines[startLine:] {
		values := strings.Split(line, ",")
		if len(values) < 4 {
			log.Warn("Unexpected rocm-smi format", zap.String("line", line))
			continue
		}
		total, _ := strconv.Atoi(strings.TrimSpace(values[2]))
		used, _ := strconv.Atoi(strings.TrimSpace(values[3]))
		gpuStat := GPUStat{
			Name:          strings.TrimSpace(values[1]),
			DriverVersion: "N/A",
			VRAMTotalMB:   total,
			VRAMUsedMB:    used,
		}
		stats = append(stats, gpuStat)
	}
	return stats, nil
}

func parseMemoryString(memStr string) (int, error) {
	re := regexp.MustCompile(`(\d+)\s*(MB|GB|TB)`)
	matches := re.FindStringSubmatch(memStr)
	if len(matches) < 3 {
		val, err := strconv.Atoi(strings.TrimSpace(memStr))
		if err == nil {
			return val, nil
		}
		return 0, fmt.Errorf("invalid memory string format: %s", memStr)
	}

	val, _ := strconv.Atoi(matches[1])
	unit := strings.ToUpper(matches[2])
	switch unit {
	case "GB":
		val *= 1024
	case "TB":
		val *= 1024 * 1024
	}
	return val, nil
}
