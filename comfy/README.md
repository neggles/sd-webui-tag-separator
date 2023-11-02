# comfyui node for tag separator

copy [`tag_separator.py`](./tag_separator.py) into `ComfyUI/custom_nodes/` folder.

This is not hugely tested and may not be kept up to date with the main extension,
since code has to be copied between them and automating that is a *chore and a half.*

You can also do this:
```sh
cd /path/to/ComfyUI/custom_nodes/
curl -fsSL -o tag_separator.py 'https://github.com/neggles/sd-webui-tag-separator/raw/main/comfy/tag_separator.py'
```
to download it directly on Linux. on Windows, you can use `Invoke-WebRequest` in PowerShell.
