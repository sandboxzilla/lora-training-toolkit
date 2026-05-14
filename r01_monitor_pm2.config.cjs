'use strict';

module.exports = {
  apps: [
    {
      name: 'r01-monitor',
      script: '/home/erol/lora_training/monitor_r01.sh',
      interpreter: '/bin/bash',
      cwd: '/home/erol/lora_training',
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 50,
      min_uptime: '10s',
      watch: false,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/home/erol/.pm2/logs/r01-monitor-error.log',
      out_file: '/home/erol/.pm2/logs/r01-monitor-out.log',
      merge_logs: true,
    },
  ],
};
