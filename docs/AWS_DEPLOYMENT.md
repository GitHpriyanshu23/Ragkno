# Launch RagKno on AWS EC2 with Cloudflare Pages

Keep the frontend at https://ragkno.pages.dev. Run the FastAPI backend on one EC2 instance, with Caddy providing HTTPS and a Docker volume preserving Chroma and model caches. PostgreSQL remains a separately configured managed database.

## Account and cost boundary

For a new eligible AWS account, select the **Free account plan**. AWS currently provides $100 signup credits and opportunities to earn an additional $100. The free plan ends after six months or when credits are exhausted, whichever happens first. Do not upgrade to Paid unless you intend to accept charges. An instance marked “Free tier eligible” consumes credits under this program; it is not an unlimited free server.

Check Billing → Free Tier / Credits before launching. Monitor the combined usage of compute, EBS disk, public IPv4, and data transfer. AI provider usage and domain registration are separate from AWS credits.

Official references: [EC2 free-tier rules](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html), [account plans](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html).

## 1. Launch the server

In EC2 → Launch instance:

- Name: `ragkno-backend`.
- Region: Mumbai (`ap-south-1`) if the selected instance is available to your account there.
- Image: Canonical Ubuntu Server 24.04 LTS, **64-bit x86**, without paid Marketplace software.
- Instance: `m7i-flex.large` (2 vCPU, 8 GiB RAM). Confirm AWS marks it eligible for your new free plan before launch. Do not substitute a paid-only instance automatically.
- Key pair: create `ragkno-aws`, download the `.pem` and keep it locally.
- Network: default VPC, public subnet, auto-assign public IPv4 enabled.
- Security group: SSH TCP 22 from **My IP** only; HTTP TCP 80 and HTTPS TCP 443 from `0.0.0.0/0`. Keep backend port 7860 closed externally.
- Disk: 40 GiB encrypted gp3, default performance, no additional volumes. Local image builds and model caches need disk space; monitor usage.

Review the estimated usage and your account plan before clicking Launch. Avoid adding a load balancer or NAT gateway for this single-server setup.

References: [launch settings](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-launch-parameters.html), [security groups](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/security-group-rules-reference.html).

## 2. Connect and install Docker

Replace the path and IP:

```bash
chmod 400 ~/Downloads/ragkno-aws.pem
ssh -i ~/Downloads/ragkno-aws.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

On the server:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo systemctl enable --now docker
git clone https://github.com/GitHpriyanshu23/Ragkno.git
cd Ragkno
```

Confirm the deployed Git revision contains `deploy/aws/compose.yaml`. Push the prepared deployment files to GitHub before cloning, or copy those files to the server. Do not copy your personal development `.env` blindly.

## 3. Configure hostname and secrets

Create an A record for a backend hostname you control, such as `api.ragkno.xyz`, pointing to the EC2 public IPv4. For initial certificate setup, use DNS-only mode if Cloudflare manages the DNS. This example does not imply the domain has been purchased.

An automatically assigned EC2 public IPv4 can change after a stop/start. Update DNS if it changes. The Docker data volume survives container recreation on this disk, but is not a backup and will not survive deletion of the underlying disk.

```bash
cp deploy/aws/.env.example deploy/aws/.env
chmod 600 deploy/aws/.env
openssl rand -hex 32
nano deploy/aws/.env
```

Put the generated secret in `RAGKNO_SESSION_SECRET`, set `BACKEND_DOMAIN` to the real hostname, and fill the database/provider credentials. Preserve the Pages frontend URLs until you intentionally move the frontend to a custom domain. For Google OAuth, register both callback URLs in the example with your Google web client.

The `.env` is ignored by Git. Caddy requires the hostname to resolve to this server and ports 80/443 to be reachable to obtain and renew its TLS certificate.

## 4. Build and run

From the repository root:

```bash
sudo docker compose --env-file deploy/aws/.env -f deploy/aws/compose.yaml config --quiet
sudo docker compose --env-file deploy/aws/.env -f deploy/aws/compose.yaml up -d --build
sudo docker compose --env-file deploy/aws/.env -f deploy/aws/compose.yaml ps
sudo docker compose --env-file deploy/aws/.env -f deploy/aws/compose.yaml logs --tail=100 backend caddy
```

The initial build downloads Python packages, and the first readiness check can download models. Allow time for both. One Uvicorn worker avoids duplicating model memory.

```bash
curl -fsS https://YOUR_BACKEND_HOSTNAME/health/live
curl -fsS https://YOUR_BACKEND_HOSTNAME/health/ready
sudo docker stats --no-stream
df -h
```

These files have not yet been verified on a live EC2 instance. Check build logs and actual memory use before inviting users.

## 5. Connect the deployed frontend

In Cloudflare Pages → ragkno → Settings → Variables and Secrets, set the **Production runtime** binding:

```env
BACKEND_ORIGIN=https://YOUR_BACKEND_HOSTNAME
```

No `/api` suffix. Keep the frontend build variable `VITE_API_URL=/api`. Redeploy Pages to apply the binding.

```bash
curl -fsS https://ragkno.pages.dev/api/health/live
curl -fsS https://ragkno.pages.dev/api/health/ready
```

Test registration/login, a small upload, streamed answers and citations. Restart the backend container and confirm the indexed source still answers questions. Test Google sign-in and Drive if enabled. Back up the database and backend data independently before launch.

## Updating and stopping

For a reviewed update, pull the intended Git revision and rerun the build/start command. Keep the previous revision available for rollback. Do not run `docker compose down -v` during updates: it deletes the persistent volumes.

Stopping EC2 stops compute usage, but retained EBS storage and some other resources can continue consuming credits. Monitor Billing, and plan a migration or paid budget before the free plan expires.
