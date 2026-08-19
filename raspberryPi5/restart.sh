#!/bin/bash
set -e

systemctl restart detect-api.service
systemctl status detect-api.service --no-pager -l | head -20
