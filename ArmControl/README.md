安装nginx
```
apt-get install nginx -y
cp dist /home/jetson/Desktop
cp nginx.conf /etc/nginx
rm -rf /etc/nginx/sites-enabled/default
systemctl enable nginx
systemctl start nginx
```

安装
```
cp api.py run.sh /home/jetson/Desktop
chmod +x /home/jeston/Desktop/run.sh
cp api.service /etc/systemd/system/
systemctl start api
systemctl enable api
```
