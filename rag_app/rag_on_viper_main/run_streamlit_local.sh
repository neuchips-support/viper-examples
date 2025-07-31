#!/bin/bash

# Get the current working directory
current_dir=$(pwd)

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Ensure the correct conda script is sourced
conda activate neullamacpp

# Check if the environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment 'neullamacpp'. Please check if it exists."
    exit 1
fi

# Wait for the environment activation to stabilize
sleep 1

#!/bin/bash

# Check if any Streamlit process is running
streamlit_pid=$(pgrep -f "streamlit run")

if [ ! -z "$streamlit_pid" ]; then
    echo "Found Streamlit process with PID: $streamlit_pid"
    
    # Kill the existing Streamlit process
    kill -9 $streamlit_pid
    echo "Killed Streamlit process with PID: $streamlit_pid"

    # Check if the process is still running
    sleep 2  # Give it a couple of seconds to terminate
    if ps -p $streamlit_pid > /dev/null; then
        echo "Streamlit process $streamlit_pid is still running. Kill failed."
        exit 1
    else
        echo "Streamlit process $streamlit_pid successfully killed."
    fi
fi

# Now start the new Streamlit app in the background
LANGUAGE=${1:-zh-CN}
echo $LANGUAGE

echo "Starting new Streamlit app..."
streamlit run "$current_dir/neuTorch_main.py" -- --language "$LANGUAGE"

