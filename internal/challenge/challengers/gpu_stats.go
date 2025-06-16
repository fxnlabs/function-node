package challengers

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"

	"go.uber.org/zap"
)

// GPUStatsChallenger polls GPU stats.
type GPUStatsChallenger struct{}

// Execute polls the metadata of the GPUs using nvidia-smi.
func (c *GPUStatsChallenger) Execute(payload interface{}, log *zap.Logger) (interface{}, error) {
	log.Info("Polling GPU stats...")

	var output []byte
	var err error

	var gpuType string
	switch runtime.GOOS {
	case "darwin":
		output, err = getMacGPUStats(log)
		gpuType = "mac"
	case "linux":
		// For Linux, try nvidia-smi first, then rocm-smi
		output, err = getNvidiaGPUStats(log)
		gpuType = "nvidia"
		if err != nil {
			log.Warn("nvidia-smi failed, trying rocm-smi", zap.Error(err))
			output, err = getAmdGPUStats(log)
			gpuType = "amd"
		}
	default:
		// Fallback to nvidia-smi for other OSes
		output, err = getNvidiaGPUStats(log)
		gpuType = "nvidia"
	}

	if err != nil {
		log.Error("Failed to get GPU stats", zap.Error(err))
		return nil, err
	}

	stats, err := parseGPUStats(output, gpuType, log)
	if err != nil {
		log.Error("Failed to parse GPU stats", zap.Error(err))
		return nil, err
	}

	log.Info("Successfully polled GPU stats")
	return stats, nil
}

func getMacGPUStats(log *zap.Logger) ([]byte, error) {
	cmd := exec.Command("system_profiler", "SPDisplaysDataType")
	output, err := cmd.Output()
	if err != nil {
		log.Error("system_profiler failed", zap.Error(err))
		return nil, err
	}
	return output, nil
}

func getNvidiaGPUStats(log *zap.Logger) ([]byte, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			log.Error("nvidia-smi failed", zap.Error(exitErr), zap.String("stderr", string(exitErr.Stderr)))
			// nvidia-smi not found, handle this case gracefully
			if strings.Contains(string(exitErr.Stderr), "executable file not found") || strings.Contains(string(exitErr.Stderr), "No such file or directory") {
				log.Warn("nvidia-smi command not found, skipping GPU stats poll")
				return nil, fmt.Errorf("nvidia-smi not found")
			}
		}
		log.Error("Failed to execute nvidia-smi", zap.Error(err))
		return nil, err
	}
	return output, nil
}

func getAmdGPUStats(log *zap.Logger) ([]byte, error) {
	cmd := exec.Command("rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv")
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			log.Error("rocm-smi failed", zap.Error(exitErr), zap.String("stderr", string(exitErr.Stderr)))
			if strings.Contains(string(exitErr.Stderr), "executable file not found") || strings.Contains(string(exitErr.Stderr), "No such file or directory") {
				log.Warn("rocm-smi command not found, skipping GPU stats poll")
				return nil, fmt.Errorf("rocm-smi not found")
			}
		}
		log.Error("Failed to execute rocm-smi", zap.Error(err))
		return nil, err
	}
	return output, nil
}

func parseGPUStats(output []byte, gpuType string, log *zap.Logger) (map[string]interface{}, error) {
	switch gpuType {
	case "mac":
		return parseMacGPUStats(output, log)
	case "nvidia":
		return parseNvidiaGPUStats(output, log)
	case "amd":
		return parseAmdGPUStats(output, log)
	default:
		return nil, fmt.Errorf("unsupported gpu type: %s", gpuType)
	}
}

func parseMacGPUStats(output []byte, log *zap.Logger) (map[string]interface{}, error) {
	stats := make(map[string]interface{})
	lines := strings.Split(string(output), "\n")
	var currentGPU map[string]interface{}
	gpuIndex := 0

	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "Chipset Model:") {
			if currentGPU != nil {
				stats[fmt.Sprintf("gpu_%d", gpuIndex)] = currentGPU
				gpuIndex++
			}
			currentGPU = make(map[string]interface{})
			currentGPU["name"] = strings.TrimSpace(strings.Split(trimmedLine, ":")[1])
		} else if strings.HasPrefix(trimmedLine, "VRAM (Total):") && currentGPU != nil {
			currentGPU["memory_total"] = strings.TrimSpace(strings.Split(trimmedLine, ":")[1])
		}
	}

	if currentGPU != nil {
		stats[fmt.Sprintf("gpu_%d", gpuIndex)] = currentGPU
	}

	return stats, nil
}

func parseNvidiaGPUStats(output []byte, log *zap.Logger) (map[string]interface{}, error) {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	stats := make(map[string]interface{})
	for i, line := range lines {
		values := strings.Split(line, ", ")
		if len(values) < 7 {
			log.Warn("Unexpected nvidia-smi format", zap.String("line", line))
			continue
		}
		gpuStats := map[string]interface{}{
			"name":                values[0],
			"driver_version":      values[1],
			"memory_total_mb":     values[2],
			"memory_used_mb":      values[3],
			"memory_free_mb":      values[4],
			"utilization_gpu_pct": values[5],
			"utilization_mem_pct": values[6],
		}
		stats[fmt.Sprintf("gpu_%d", i)] = gpuStats
	}
	return stats, nil
}

func parseAmdGPUStats(output []byte, log *zap.Logger) (map[string]interface{}, error) {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	stats := make(map[string]interface{})
	// rocm-smi output has a header, so we skip it.
	for i, line := range lines[1:] {
		values := strings.Split(line, ",")
		if len(values) < 5 {
			log.Warn("Unexpected rocm-smi format", zap.String("line", line))
			continue
		}
		gpuStats := map[string]interface{}{
			"name":            values[1],
			"driver_version":  "N/A",
			"memory_total_mb": values[2],
			"memory_used_mb":  values[3],
		}
		stats[fmt.Sprintf("gpu_%d", i)] = gpuStats
	}
	return stats, nil
}
