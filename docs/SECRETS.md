# Credentials

**macOS Keychain is the source of truth.** Nothing secret lives in this repo,
and `.gitignore` covers `secrets.env`, `.env*`, and `*.pem` as a backstop.

```bash
python3 -m racewm.secrets                      # what is set, and from where
python3 -m racewm.secrets --set WANDB_API_KEY  # rotate (input hidden)
python3 -m racewm.secrets --sync-backup        # rewrite the offline backup
python3 -m racewm.secrets --aws-profile        # rewrite the aws-cli profile
```

## Resolution order

| # | source | why |
|---|---|---|
| 1 | process environment | CI, containers, one-off overrides |
| 2 | **macOS Keychain**, service `race-wm:<KEY>` | primary store on this machine |
| 3 | `~/Documents/dev/secrets/race-wm.env` (0600, dir 0700) | offline backup, non-macOS boxes |
| 4 | `~/Documents/dev/sunnypilot/secrets.yaml` | legacy: RunPod + Upstash only |

Read one by hand:

```bash
security find-generic-password -a "$USER" -s "race-wm:S3_BUCKET" -w
```

`racewm/secrets.py` never parses `~/.netrc` or `~/.aws/credentials` — `wandb`
and `boto3` read those themselves, so copying the values into our process
would add exposure and buy nothing. `describe()` reports key names, source, and
length; never a value.

## AWS

IAM user **`race-wm`**, inline policy `race-wm-s3-scoped`:

```
s3:ListBucket, s3:GetBucketLocation      on arn:aws:s3:::puffertank
s3:GetObject, s3:PutObject, s3:DeleteObject,
s3:AbortMultipartUpload, s3:ListMultipartUploadParts
                                          on arn:aws:s3:::puffertank/*
```

Verified by exercise, not by reading the policy:

```
ALLOW  put/get/list/delete under s3://puffertank/race-wm/
DENY   anything in s3://drift-datasets/     DENY  ListAllMyBuckets
DENY   CreateBucket                         DENY  iam:CreateUser
```

Bucket `puffertank` (us-west-1): all public access blocked, SSE-AES256 at rest,
versioning off — shards are large and deterministically regenerable, so version
history would only double storage cost.

aws-cli profile **`race-wm`**. The `[default]` profile is deliberately left
alone.

```bash
aws s3 ls s3://puffertank/ --profile race-wm
```

## Known issues on this machine

**The `[default]` aws-cli profile is invalid.** Its `aws_access_key_id` is
`user_35P…`, which is not an AWS key format (AWS keys begin `AKIA`/`ASIA`) —
some other service's token was written there. Pre-existing, unrelated to this
project, and worth cleaning up separately.

**Root access keys still exist on the account.** Root MFA is enabled, which is
the protection that matters most, and this project no longer uses the root key.
Before deleting it, give the drift work its own scoped IAM user, or that
tooling breaks. Sequence:

1. create IAM user `drift`, scoped to `s3://drift-datasets/*`
2. point whatever drift uses at that key
3. verify drift tooling still works
4. only then delete the root access key in the IAM console

## Rotating the wandb key

wandb has **no programmatic key management** — `api_key` only reads the current
value, and the CLI has `login` but no rotation. It must be done in the browser:

1. https://wandb.ai/authorize — revoke the old key, create a new one
2. `python3 -m racewm.secrets --set WANDB_API_KEY`

The entity is a team (`drift-ai-models`), so a **service-account** key
(Team Settings → Service Accounts) is preferable to a personal key: a personal
key acts as you across every project you belong to, and dies whenever you
rotate it.
