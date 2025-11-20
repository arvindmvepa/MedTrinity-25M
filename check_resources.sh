#!/bin/bash
# check_system_resources.sh

echo "=== SYSTEM RESOURCES ==="
echo "CPU Cores (Physical): $(nproc --all)"
echo "CPU Threads (Logical): $(grep -c ^processor /proc/cpuinfo)"
echo "Memory Total: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Memory Available: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"

echo -e "\n=== RECOMMENDED SETTINGS ==="
CORES=$(nproc --all)
MEM_GB=$(free -g | grep '^Mem:' | awk '{print $2}')

# Conservative recommendation: use 75% of cores, ensure at least 2GB per process
RECOMMENDED_CORES=$((CORES * 3 / 4))
MAX_BY_MEMORY=$((MEM_GB / 2))

if [ $MAX_BY_MEMORY -lt $RECOMMENDED_CORES ]; then
    RECOMMENDED=$MAX_BY_MEMORY
else
    RECOMMENDED=$RECOMMENDED_CORES
fi

echo "Conservative (75% cores): $RECOMMENDED_CORES"
echo "Memory-limited (2GB/proc): $MAX_BY_MEMORY" 
echo "Recommended: $RECOMMENDED"
echo -e "\nUsage: python assign_brats_goat_to_gli_and_met.py --n_jobs $RECOMMENDED"