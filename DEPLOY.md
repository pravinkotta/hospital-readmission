# Deploying to EC2

The container built by `Dockerfile` runs the same API locally and on AWS.
Nothing changes except `--host 0.0.0.0` and an open port.

> **Cost control:** an instance bills while it runs. Stop it between interview
> sessions (`aws ec2 stop-instances`) and start it before — an attached
> **Elastic IP** keeps the address stable and costs nothing while the instance
> runs.

---

## 1. Launch (once)

1. EC2 → **Launch instance** → name `readmission-api`
2. AMI: **Amazon Linux 2023** (what this walkthrough was executed on) — note
   the default SSH user is **`ec2-user`**, not `ubuntu`. An Ubuntu AMI works
   too; swap the user to `ubuntu` and `dnf` to `apt`/`get.docker.com` below.
3. Instance type: **t3.micro** (free tier, 1 vCPU / 1 GB — enough to *serve*;
   the build needs the swap from step 2). `t3.small` if you prefer headroom.
4. Key pair: create/download `.pem`, keep it private (see the Windows note
   below)
5. **Security group** — inbound rules:
   | Type | Port | Source |
   |---|---|---|
   | SSH | 22 | your IP (`/32`), never `0.0.0.0/0` |
   | Custom TCP | 8000 | your IP for a private demo, or `0.0.0.0/0` for a public one |
6. **Elastic IPs** → allocate → associate with the instance (survives stops)

> **Windows `.pem` ACL trap** (cost an hour here): OpenSSH refuses keys whose
> ACL lists *any* principal besides you — including same-named groups and
> inherited SYSTEM/Administrators ACEs. `icacls ... /inheritance:r /grant:r`
> alone may not clear them. Nuclear option that works:
>
> ```powershell
> icacls $f /inheritance:r
> icacls $f /remove:g "pravin\"  # the odd group ACE — your domain\ may differ
> icacls $f /remove:g "NT AUTHORITY\SYSTEM" "BUILTIN\Administrators"
> icacls $f /grant:r "$env:USERNAME\$env:USERNAME:(R)"
> # expected result: exactly one line —  <user>:R
> ```
>
> Symptom if wrong: `Bad permissions ... key will be ignored`, then
> `Permission denied (publickey)`.

## 2. First setup (SSH once)

```bash
ssh -i readmission.pem ec2-user@<PUBLIC_IP>

# Docker (Amazon Linux 2023 — dnf; Ubuntu would be: curl -fsSL https://get.docker.com | sh)
sudo dnf -y install docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user     # new SSH sessions pick the group up

# 2 GB swap — pip installing torch on a 1 GB instance OOMs without it
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 3. Ship the code

```bash
# from your machine, in the repo root
scp -i readmission.pem -r src artifacts requirements.txt Dockerfile .dockerignore ec2-user@<PUBLIC_IP>:~/app/
scp -i readmission.pem data/diabetic_data.csv data/IDS_mapping.csv ec2-user@<PUBLIC_IP>:~/app/
ssh -i readmission.pem ec2-user@<PUBLIC_IP> "mkdir -p ~/app/data && mv ~/app/*.csv ~/app/data/"
# or, once the repo is on GitHub:
#   git clone <repo-url> && cd <repo>
```

The `.env` is copied **separately and explicitly** — it is gitignored and must
never be committed. It must use `KEY=value` lines (Docker's `--env-file`
cannot parse a bare `sk-...` line):

```bash
scp -i readmission.pem .env ec2-user@<PUBLIC_IP>:~/app/.env
```

## 4. Build & run

```bash
ssh -i readmission.pem ec2-user@<PUBLIC_IP>
cd ~/app
docker build -t readmission-api .        # ~10 min on a micro; sudo if not in the docker group yet
docker run -d --name api --restart unless-stopped -p 8000:8000 --env-file .env readmission-api
```

`--restart unless-stopped` is the "it survives a reboot / a crash" requirement;
a container that silently dies before an interview is the actual risk here.

## 5. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"ReadmissionMLP 120->64->32->1","threshold":0.57,...}

# from your machine (public):
curl http://<ELASTIC_IP>:8000/health
curl http://<ELASTIC_IP>:8000/cohort/sample
curl -X POST http://<ELASTIC_IP>:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"patient_nbr": 88479036, "explain": true}'

# the spendy endpoint — note the header
curl -X POST http://<ELASTIC_IP>:8000/alert \
     -H "Content-Type: application/json" -H "X-API-Key: $ALERT_API_KEY" \
     -d '{"patient_nbr": 88479036}'
```

## 6. Logs / stop / start

```bash
docker logs -f api                # follow the service
aws ec2 stop-instances   --instance-ids i-0...    # stop billing (Elastic IP stays)
aws ec2 start-instances  --instance-ids i-0...    # before the interview
```

---

## Security notes (the ones interviewers ask about)

- **`/alert` costs money per call** — it is gated by `X-API-Key` and the server
  refuses to run it without `ALERT_API_KEY` configured (503). Rotate the key
  after the demo (`ALERT_API_KEY` in `.env` → restart container).
- The API key check uses constant-time comparison (`secrets.compare_digest`).
- No port 80/443, no TLS: for a portfolio demo over HTTP on a non-standard
  port this is acceptable; a real deployment terminates TLS at a load balancer
  (ALB) or reverse proxy, which is out of scope here (C8).
- `.dockerignore` excludes `.env` — the secret is *injected at run time*
  (`--env-file`), never baked into an image layer.
