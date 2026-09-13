#!/bin/bash
# Ensure TI mmWave radar USB serial devices are available in the container

set -e

echo "Checking for TI mmWave radar USB devices..."

# Wait up to 10 seconds for devices to appear
for i in {1..10}; do
    if [ -e /dev/ttyUSB0 ] && [ -e /dev/ttyUSB1 ]; then
        echo "✓ Found ttyUSB0 and ttyUSB1"
        chmod 666 /dev/ttyUSB0 /dev/ttyUSB1 2>/dev/null || true
        ls -la /dev/ttyUSB* 2>/dev/null || true
        exit 0
    fi
    
    # Try to create device nodes if they don't exist
    if [ ! -e /dev/ttyUSB0 ]; then
        echo "Creating /dev/ttyUSB0..."
        mknod /dev/ttyUSB0 c 188 0 2>/dev/null || true
        chmod 666 /dev/ttyUSB0 2>/dev/null || true
    fi
    
    if [ ! -e /dev/ttyUSB1 ]; then
        echo "Creating /dev/ttyUSB1..."
        mknod /dev/ttyUSB1 c 188 1 2>/dev/null || true
        chmod 666 /dev/ttyUSB1 2>/dev/null || true
    fi
    
    if [ -e /dev/ttyUSB0 ] && [ -e /dev/ttyUSB1 ]; then
        echo "✓ USB devices ready"
        ls -la /dev/ttyUSB* 2>/dev/null || true
        exit 0
    fi
    
    echo "Waiting for USB devices... ($i/10)"
    sleep 1
done

echo "⚠ Warning: USB devices not found. Make sure radar is connected."
echo "   You can run this script again after plugging in the radar:"
echo "   /workspace/scripts/setup_usb_devices.sh"
exit 1
