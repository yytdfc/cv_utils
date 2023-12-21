<p align="center"><img src="assets/fastcv.webp" height=120></p>

## Tenets

- the most common type is numpy
- size always be height, width

.
├── core
│   ├── io
│   ├── view
│   ├── cast
│   ├── auto




### layout

- np: hwc
- pil: hwc
- cv2: hwc
- tensor: nchw


### cast

```
# type: tensor, numpy, pil, cv2
# dtype: auto, fp32, fp16
# range: [0, 255], [-1.0, 1.0], [0.0, 1.0
fastcv.cast(t, type="tensor", )
```
