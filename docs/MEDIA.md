# Image and video requests

Start with `--media`. This loads the NVFP4 checkpoint's vision encoder and the existing validated vision recipe (BF16/auto KV,3GiB KV,16384 internal context, prefix cache enabled,128MiB processor cache, query32). Text mode keeps its FP8 KV and compact decision head/MoE optimizations. Media mode deliberately does not inherit those text-only optimizations without validation.

The text contract stays unchanged. Add:

```json
{"media":{"type":"image","data":"data:image/png;base64,..."}}
```

For video use `type: "video"`, `data:video/mp4;base64,...`. One image OR one video per request; every question uses that same media. `state` remains a required nonempty text instruction/context. Never put a file path or HTTP URL in `data`. No external URL fetching or caller-controlled server file access is allowed.

Limits:8MiB HTTP/CLI JSON body;6MiB encoded media; PNG/JPEG <=1920 pixels per axis and2073600 total pixels; MP4 <=1280x720,10 seconds,300 source frames. These are acceptance/resource limits, not accuracy guarantees. Animated images and audio inputs are rejected. Expanded text+media tokens must fit `--max-input-tokens` (1..8192); no silent text truncation. All questions preprocess before any inference, so an oversized later question cannot cause earlier GPU judgments to be partially executed.

Video uses the pinned vLLM processor's frame sampling. **It does not inspect every source frame or listen to audio.** Output reports source frames/duration and media token counts per question, not a claim of full-frame coverage. Validation includes4-second,4-frame,1fps clips; longer/dense real-world clips remain unverified. Image resizing/tokenization follows the pinned processor. Reference regression checks compare full expanded token IDs to detect unintended changes.

`examples/image-request.json` and `video-request.json` embed project-authored geometric fixtures. To build a request for your own file locally:

```sh
python3 scripts/media-request.py image /path/image.png examples/request.json > my-image.json
python3 scripts/media-request.py video /path/clip.mp4 examples/request.json > my-video.json
```

The client helper reads a local file explicitly chosen by you; the server itself does not open request-provided paths. Adapt the questions/state in your template. The output distributions are uncalibrated.

The v0.3.0 extension applies to text mode only. `--media` retains the previous8192expanded-input limit,16384internal context and3GiB KV. No image/video limit or audio support has been added.
