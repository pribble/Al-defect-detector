source ./tool/exportPATH.sh
cd ssd_detection/ssd_detection_src
bash ./build.sh
cp ./build/ssd_detection ../../paddle_frame/
