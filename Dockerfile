# Reproducible training/inference environment for ck3dna.
#
# Rebuilt from Hakim1625/ck3-image-portrait-modeling's original Dockerfile
# pattern (same pytorch-lightning base image, git + git-lfs for pulling
# large files), pointed at this repo instead.
FROM pytorchlightning/pytorch_lightning:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        cmake \
        build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone this repo instead of copying, mirroring the upstream project's
# self-contained, git-lfs-aware container pattern.
RUN git clone https://github.com/amartin1213/CK3-DNA-Generator.git .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# dlib's shape predictor is not shipped in-repo (license/size); mount or
# download it at runtime -- see README.md "Getting the dlib landmark model".

ENTRYPOINT ["python"]
