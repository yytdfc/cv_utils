import numpy as np
import cv2
import ffmpeg
from ..cast import cast


class FFmpegVideoDumper:
    def __init__(
        self, path, monit=False, fps=30, codec="h264", crf=28, preset="veryfast", h=0, w=0,
    ):
        self.idx = 0
        self.path = path
        self.dumper = None
        self.monit = monit
        self.codec = codec
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.h = h
        self.w = w

    def __del__(self):
        self.close()

    def close(self):
        if self.dumper is not None:
            self.dumper.stdin.close()
            self.dumper.wait()

    def release(self):
        self.close()

    def write(self, img):
        img = cast(img, type="np")
        if self.dumper is None:
            self.idx = 0
            self.dumper = True
            if self.h == 0 or self.w==0:
                self.h, self.w, _ = img.shape
            self.h, self.w = self.h // 2 * 2, self.w // 2 * 2
            if self.codec =="h264":
                opt_dict = dict(
                    crf=self.crf,
                    pix_fmt="yuv420p",
                    s=f"{self.w}x{self.h}",
                    preset=self.preset,
                    codec="h264",
                    r=self.fps,
                )
            elif self.codec =="yuv":
                opt_dict = dict(
                    crf=self.crf,
                    pix_fmt="yuv420p",
                    s=f"{self.w}x{self.h}",
                )
            self.dumper = (
                ffmpeg.input(
                    "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{self.w}x{self.h}"
                )
                .output(
                    self.path,
                    **opt_dict,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )
        # crop odd size with 1 pix
        self.dumper.stdin.write(img[: self.h, : self.w].astype(np.uint8).tobytes())
        self.idx += 1
        if self.monit:
            cv2.imshow("moniter", img)
            if cv2.waitKey(1) == ord("q"):
                return


class FFmpegVideoLoader:
    def __init__(self, path, fps=0, resize_min_size=0):
        probe = ffmpeg.probe(path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        self.width = int(video_stream["width"])
        self.height = int(video_stream["height"])
        if resize_min_size != 0:
            scale = resize_min_size / min(self.height, self.width)
            self.width = round(self.width * scale) // 2 * 2
            self.height = round(self.height * scale) // 2 * 2
        if round(int(video_stream.get("tags", {}).get("rotate", 0)) / 90) % 2 != 0:
            self.width, self.height = self.height, self.width

        self.length = int(video_stream["nb_frames"])
        ori_fps = eval(video_stream["avg_frame_rate"])
        if fps == 0:
            self.fps = ori_fps
        else:
            self.fps = fps
            # ToDo: length here is not so robust
            self.length = round(self.length / ori_fps * self.fps)

        self.loader = (
            ffmpeg.input(path)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                r=self.fps,
                s=f"{self.width}x{self.height}",
            )
            .run_async(pipe_stdout=True, quiet=True)
        )
        self.idx = 0

    def __del__(self):
        self.close()

    def close(self):
        # reading left frames
        self.loader.stdout.close()
        self.loader.wait()

    def release(self):
        self.close()

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        img = self.read()
        if img is None:
            raise StopIteration
        else:
            return img

    def __len__(self):
        return self.length

    def get_fps(self):
        return self.fps

    def get_idx(self):
        return self.idx

    def read(self):
        in_bytes = self.loader.stdout.read(self.width * self.height * 3)
        if not in_bytes:
            return None
        img = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        self.idx += 1
        return img
