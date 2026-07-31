set EMBEDDED=C:\intelFPGA_lite\23.1std\embedded

%EMBEDDED%\host_tools\altera\device_tree\sopc2dts --input ..\soc_system.sopcinfo --output soc_system.dts --type dts --board .\soc_system_board_info.xml --board .\hps_common_board_info.xml --bridge-removal all --clocks

%EMBEDDED%\host_tools\gnu\dtc\dtc -I dts -O dtb -o soc_system.dtb soc_system.dts

pause
