安装MVS，下载地址：http://10.1.80.80/ai-system-courses/defect-detection/MVS.tgz
```
wget http://10.1.80.80/ai-system-courses/defect-detection/MVS.tgz
tar xf MVS.tgz
cd MVS-1.0.0_aarch64/
./setup.sh
```

安装需要的python库
```
pip3 install Flask Pillow pyserial -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install Cython==0.29.30 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install numpy==1.19.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install matplotlib==3.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install scikit-image==0.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装
```
cp -r Font templates api.py database.py config.ini run.sh yuanshi.jpg /opt/MVS/Samples/aarch64/Python/GrabImage
chmod +x /opt/MVS/Samples/aarch64/Python/GrabImage/run.sh
cp detect-api.servie /etc/systemd/system
```

启动服务
```
systemctl enable detect-api.service
systemctl start detect-api.service
```

## 需要有海康相机
