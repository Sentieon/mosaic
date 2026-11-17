# Local Iceberg + MinIO test stack

This directory provides the local development stack for MOSAIC: an Apache
Iceberg REST catalog backed by MinIO (S3-compatible) storage, plus a PostgreSQL
metadata store for the catalog. The pipeline's default configuration points at
these services, so no environment variables are needed for local development.

## Start the services

```bash
cd iceberg_test
docker-compose up -d
```

This starts:
- Iceberg REST catalog on port 8181
- MinIO (S3-compatible storage) on port 9000 (API) and 9001 (console)
- MinIO client (mc), which creates the `warehouse` bucket and exits
- PostgreSQL (the catalog's metadata store)

Verify with `docker-compose ps`.

If the `warehouse` bucket is missing, create it via the MinIO console
(http://localhost:9001, admin/password) or:

```bash
python3 -c "import boto3; boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id='admin', aws_secret_access_key='password').create_bucket(Bucket='warehouse')"
```

Some clients also need the `minio` hostname to resolve:

```bash
echo "127.0.0.1  minio" | sudo tee -a /etc/hosts
```

## MinIO console

- URL: http://localhost:9001
- Username: `admin`
- Password: `password`

Data is stored in the `warehouse` bucket. These credentials are the zero-config
local-dev defaults baked into `docker-compose.yml` — not secrets.

## Cleanup

```bash
docker-compose down       # stop the services
docker-compose down -v    # also delete all data (volumes)
```
