from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YukunZhou/RETFound_mae_natureCFP",
    local_dir="./weights",
    token="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"
)