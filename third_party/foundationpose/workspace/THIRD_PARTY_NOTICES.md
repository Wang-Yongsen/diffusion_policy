# Third-party sources

FoundationPose core is reused under the repository's NVIDIA Source Code License. The workspace does not copy the prior simulation's rail mesh.

- `vendor/sldprt-format-research`: https://github.com/blussyya/sldprt-format-research, MIT; pinned commit `93b49335cfcd8b47e93878a0d9935c376a86169d`. Original LICENSE is retained. The decoder extracts the attached SolidWorks file's stored display tessellation in metres. It does not recover exact trimmed curved B-Rep surfaces. Diagnostics and the original CAD are preserved.
- `vendor/nvdiffrast`: https://github.com/NVlabs/nvdiffrast. Original LICENSE is retained; exact checkout is recorded during validation.
- `vendor/pytorch3d`: https://github.com/facebookresearch/pytorch3d. Original LICENSE is retained; exact checkout is recorded during validation.
- Official weights and demonstration data are downloaded from the public Google Drive links in FoundationPose/readme.md; their source and local checksums are recorded. The example object is used for environment and algorithm checks, not for real-rail accuracy claims.
