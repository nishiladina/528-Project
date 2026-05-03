# Bluetooth Guide
Note: if you're getting the code from GitHub, you might not have to make any changes to the `sdkconfig`.

## Configure your sdkconfig
In VS Code, press `Ctrl K, Ctrl O` and open the `bluetooth/esp32_code` folder. Opening this folder directly (as opposed to cd'ing into it) will activate the ESP-IDF extension. 

Click the settings icon on the bottom of the screen to open the configuration editor. If that doesn't work: click `Ctrl + Shift + P` to open the command palette. Type "menuconfig" and select `ESP-IDF: SDK Configuration Editor (Menuconfig)`. 

In the configuration editor, go to `Component config` --> `Bluetooth`. Make sure `Bluetooth` is checked, and right below that, select `NimBLE - BLE only`. Then click `Save` at the top of the screen. This will edit your `sdkconfig` file.

## Flash the code to your ESP32
Same process as normal.

## Use the python helper files
There are a few python files in the bluetooth folder. Before using any of them you have to `pip install bleak`. 

`bluetooth_test.py`: scans for 10 seconds and prints a list of available bluetooth devices. The ESP32 should show up as "IMU-Stream".

`imu_stream.py`: connects to the ESP32 via bluetooth and prints the sensor data to the terminal. 

`plot_imu_ble.py`: connects to the ESP32 via bluetooth and displays the sensor data on a graph. 

`imu_sample_rate.py`: tests how many samples per second are transmitted over the bluetooth stream.