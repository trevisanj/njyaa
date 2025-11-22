# NJYAA as a service

## Install in droplet

1. Install `/etc/njyaa.keys` (edit keys)

```
$ ls -l /etc/njyaa.env 
-rw------- 1 root root 285 Nov 22 10:59 /etc/njyaa.env
```

2. Install `/etc/systemd/system/njyaa.service`

```
$ ls -l /etc/systemd/system/njyaa.service
-rw------- 1 root root 742 Nov 22 10:53 /etc/systemd/system/njyaa.service
```

**Note** ChatGPT said 644 was ok for `njyaa.service` but I did 600 anyway

**Note** In target host, `systemctl --version` --> `systemd 229` does not recognize `StandardOutput=append:/var/log/njyaa.log` & 
`StandardError=append:/var/log/njyaa.log` (see `man systemd.exec`), but out/err can be inspected using `journalctl` (see below.) 

3. Reload systemd:
   ```sh
   sudo systemctl daemon-reload
   ```

## Operation

### Start / enable
```sh
sudo systemctl enable --now njyaa
```

### Stop
```sh
sudo systemctl stop njyaa
```

### Restart
```sh
sudo systemctl restart njyaa
```

### Status
```sh
systemctl status njyaa
```

### Logs
```sh
journalctl -u njyaa -f
```

### Disable autostart
```sh
sudo systemctl disable njyaa
```

### Debug

```sh
sudo journalctl -u njyaa -b -n 50
```

### Check where outputs are going

```
systemctl show njyaa -p StandardOutput -p StandardError
```