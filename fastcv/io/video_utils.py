import os

import cv2
import numpy as np

from ..environment import ENV


class VideoDumper:
    def __init__(self, path, monit=False, fps=30, dump_mode="video", codec="", crf=17, quality=100):
        """
        dump_mode:
            video: direct save video in path
            png: save png frames in path
            jpg: save jpg frames in path
            png2video: save png frames first, then convert to video
        codec:
            avc1: default in macos
            mp4v: default in linux
        """
        self.path = path
        self.dumper = None
        assert dump_mode in {"video", "png", "jpg", "png2video"}
        self.dump_mode = dump_mode
        self.monit = monit
        if codec == "":
            if ENV.is_mac():
                self.codec = "avc1"
            else:
                self.codec = "mp4v"
        else:
            self.codec = codec
        self.fps = fps
        self.crf = crf
        self.quality = quality

    def __del__(self):
        if self.dumper is None:
            return
        if self.dump_mode == "video":
            del self.dumper
            if ENV.is_linux() and self.codec == "h264":
                os.system(
                    f"ffmpeg -v quiet -i {self.path} -codec h264 -crf {self.crf} -profile:v high -pix_fmt yuv420p -r {self.fps} -y {ENV.tmp_dir}/{self.path}.mp4"
                )
                os.system(f"mv {ENV.tmp_dir}/{self.path}.mp4 {self.path}")
        elif self.dump_mode == "png2video":
            os.system(
                f"ffmpeg -v quiet -i {self.dir}/%05d.{self.surfix} -codec h264 -crf {self.crf} -profile:v high -pix_fmt yuv420p -r {self.fps} -y {self.path}"
            )
            os.system(f"rm -rf {self.dir}")

    def write(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.dumper is None:
            if self.dump_mode.startswith("video"):
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.dumper = cv2.VideoWriter(
                    self.path, fourcc, self.fps, img.shape[1::-1]
                )
                self.dumper.set(cv2.VIDEOWRITER_PROP_QUALITY, self.quality)
            else:
                if self.dump_mode.endswith("video"):
                    self.dir = f"{ENV.tmp_dir}/{self.path}.dir"
                else:
                    self.dir = self.path
                self.surfix = self.dump_mode[:3]
                self.surfix = self.dump_mode[:3]
                os.system(f"mkdir -p {self.dir}")
                self.idx = 0
                self.dumper = True

        if self.dump_mode.startswith("video"):
            self.dumper.write(img)
        else:
            cv2.imwrite(
                f"{self.dir}/{self.idx:05d}.{self.surfix}",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
            )
            self.idx += 1
        if self.monit:
            cv2.imshow("moniter", img)
            if cv2.waitKey(1) == ord("q"):
                return


class VideoLoader:
    def __init__(self, path):
        self.path = path
        self.loader = cv2.VideoCapture(self.path)
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        self.loader.set(cv2.CAP_PROP_POS_FRAMES, self.idx)
        return self

    def __next__(self):
        img = self.read()
        if img is None:
            raise StopIteration
        else:
            self.idx += 1
            return img

    def __len__(self):
        return int(self.loader.get(cv2.CAP_PROP_FRAME_COUNT))

    def seek(self, idx):
        self.loader.set(cv2.CAP_PROP_POS_FRAMES, idx)
        return self.read()

    def random_pick(self):
        idx = np.random.randint(0, self.__len__())
        return self.seek(idx)

    def read(self):
        if self.loader is None:
            self.loader = cv2.VideoCapture(self.path)
        ret, img = self.loader.read()
        if ret:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return None
