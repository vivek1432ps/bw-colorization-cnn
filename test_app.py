"""Tests exercise the published model, UI rendering and bounded input handling."""
import unittest
from io import BytesIO

import numpy as np
from PIL import Image
import torch

from app import SAMPLE, infer, page, unpack
from core import MAX_UPLOAD, colorize, load_model, read_image


class LumaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = load_model()
        cls.photo = SAMPLE.read_bytes()

    def test_real_checkpoint_and_png(self):
        gray, color, elapsed = infer(*self.model,self.photo,100)
        self.assertGreater(elapsed,0)
        self.assertNotEqual(gray,color)
        with Image.open(BytesIO(gray)) as before, Image.open(BytesIO(color)) as after:
            self.assertEqual(before.size,after.size)
            p = np.array(after).astype(float)
            self.assertGreater(np.mean(np.abs(p[:,:,0]-p[:,:,2])), .5)
        ui=page((gray,color,elapsed))
        self.assertIn('alt="Grayscale input"',ui)
        self.assertIn('Download colorized PNG',ui)

    def test_zero_intensity_identical_and_aspect_preserved(self):
        pic=read_image(self.photo).resize((315,137))
        neutral,color=colorize(*self.model,pic,0)
        self.assertEqual(color.shape,(137,315,3))
        np.testing.assert_array_equal(neutral,color)

    def test_reject_oversize_invalid_and_unsupported(self):
        with self.assertRaisesRegex(ValueError,"8 MB"):
            read_image(b"x"*(MAX_UPLOAD+1))
        with self.assertRaisesRegex(ValueError,"decode"):
            read_image(b"not an image")
        pic=BytesIO()
        Image.new("RGB",(2,2)).save(pic,format="BMP")
        with self.assertRaisesRegex(ValueError,"PNG, JPEG"):
            read_image(pic.getvalue())

    def test_upload_fixture(self):
        fields=(b'--demo\r\nContent-Disposition: form-data; name="intensity"\r\n\r\n125\r\n'
                b'--demo\r\nContent-Disposition: form-data; name="photo"; filename="sample.jpg"\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+self.photo+b'\r\n--demo--\r\n')
        self.assertEqual(unpack(fields,"multipart/form-data; boundary=demo"),(self.photo,125))
        with self.assertRaisesRegex(ValueError,"0 to 150"):
            unpack(fields.replace(b"125",b"155",1),"multipart/form-data; boundary=demo")

if __name__=="__main__":
    unittest.main()
